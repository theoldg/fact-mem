import os
import numpy as np
import subprocess
import shutil
import time
import multiprocessing as mp

def process_batch(start_shard, end_shard, repo_src, npy_path, output_base_dir):
    batch_name = f"shard_{start_shard}_{end_shard-1}"
    save_dir = os.path.join(output_base_dir, batch_name)
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"=== Processing {batch_name} ===")
    start_time = time.time()
    
    # 1. Load numpy file and extract slice
    print("Loading numpy file (memmap)...")
    # The file shape is (95, 1024000, 2049) as specified by user
    # We read it as raw bytes or with shape.
    # Since we want to stream it, we can just load the specific shards!
    # np.memmap allows specifying offset and shape for the view!
    
    element_size = 2 # uint16
    seq_len = 2049
    batch_size = 1024000
    shard_size_bytes = batch_size * seq_len * element_size
    
    offset = start_shard * shard_size_bytes
    shape = (end_shard - start_shard, batch_size, seq_len)
    
    # Load only the requested shards
    data = np.memmap(npy_path, dtype=np.uint16, mode='r', offset=offset, shape=shape)
    
    # 2. Write to text_data.sdsl
    ds_path = os.path.join(save_dir, "text_data.sdsl")
    print(f"Writing {ds_path} ...")
    
    with open(ds_path, "wb") as f:
        # Write placeholder header
        f.write(np.array([0], dtype=np.uint64).view(np.uint8).tobytes())
        
        # Stream data to file to avoid huge memory usage
        # We can iterate over shards and write them
        for i in range(data.shape[0]):
            print(f"  Writing shard {start_shard + i} ...")
            # Flatten the shard and convert to bytes
            f.write(data[i].tobytes())
            
        # Write terminator
        f.write(b'\xfa')
        
        # Calculate size
        ds_size = f.tell() - 8
        # Pad to 8 bytes
        if ds_size % 8 != 0:
            f.write(b'\00' * (8 - ds_size % 8))
            
        # Overwrite header with bit size
        f.seek(0)
        f.write(np.array([ds_size * 8], dtype=np.uint64).view(np.uint8).tobytes())
        
    # 3. Create dummy text_meta.sdsl
    mt_path = os.path.join(save_dir, "text_meta.sdsl")
    with open(mt_path, "wb") as f:
        f.write(np.array([0], dtype=np.uint64).view(np.uint8).tobytes())
        f.write(b'\xfa')
        mt_size = f.tell() - 8
        if mt_size % 8 != 0:
            f.write(b'\00' * (8 - mt_size % 8))
        f.seek(0)
        f.write(np.array([mt_size * 8], dtype=np.uint64).view(np.uint8).tobytes())
        
    # 4. Create dummy data_offset and meta_offset
    od_path = os.path.join(save_dir, "data_offset")
    with open(od_path, "wb") as f:
        f.write(np.array([0], dtype=np.uint64).view(np.uint8).tobytes())
        
    om_path = os.path.join(save_dir, "meta_offset")
    with open(om_path, "wb") as f:
        f.write(np.array([0], dtype=np.uint64).view(np.uint8).tobytes())
        
    # 5. Run indexing binaries
    rust_bin = os.path.join(repo_src, "rust_indexing")
    cpp_bin = os.path.join(repo_src, "cpp_indexing")
    
    parts_dir = os.path.join(save_dir, "parts")
    merged_dir = os.path.join(save_dir, "merged")
    bwt_dir = os.path.join(save_dir, "bwt")
    
    os.makedirs(parts_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)
    os.makedirs(bwt_dir, exist_ok=True)
    
    # We use all available CPUs
    cpus = mp.cpu_count()
    print(f"Using {cpus} CPUs for indexing.")
    
    # Calculate ratio (40-bit address usually enough for file > 4GB)
    ratio = int(np.ceil(np.log2(ds_size) / 8))
    print(f"Calculated ratio: {ratio}")
    
    # Run rust_indexing make-part
    # We need to split into parts to allow parallel construction!
    # In indexing.py it splits based on memory limit.
    # Let's use a simple splitting: divide into equal parts based on cpus!
    
    DS_OFFSET = 8
    HACK = 100000
    
    mem_bytes = 150 * 1024**3
    num_job_batches = 1
    while num_job_batches * (mem_bytes // 12) < ds_size:
        num_job_batches *= 2
        
    total_jobs = num_job_batches * cpus
    S = ds_size // total_jobs
    
    print(f"Using {num_job_batches} batches of {cpus} jobs each, for a total of {total_jobs} jobs.")
    
    print("Running rust_indexing make-part ...")
    for batch_start in range(0, total_jobs, cpus):
        batch_end = min(batch_start + cpus, total_jobs)
        batch_ranges = []
        for i in range(batch_start, batch_end):
            s, e = DS_OFFSET + i * S, DS_OFFSET + min((i + 1) * S + HACK, ds_size)
            batch_ranges.append((s, e))
            
        pipes = []
        for (s, e) in batch_ranges:
            cmd = [
                rust_bin, "make-part",
                "--data-file", os.path.abspath(ds_path),
                "--parts-dir", os.path.abspath(parts_dir),
                "--start-byte", str(s),
                "--end-byte", str(e),
                "--ratio", str(ratio)
            ]
            pipes.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
            
        # Wait for this batch to finish
        for pipe in pipes:
            pipe.wait()
            if pipe.returncode != 0:
                print(f"Error in make-part: {pipe.stderr.read().decode()}")
                raise RuntimeError("make-part failed")
        
    # Run rust_indexing merge
    print("Running rust_indexing merge ...")
    cmd = [
        rust_bin, "merge",
        "--data-file", os.path.abspath(ds_path),
        "--parts-dir", os.path.abspath(parts_dir),
        "--merged-dir", os.path.abspath(merged_dir),
        "--bwt-dir", os.path.abspath(bwt_dir),
        "--num-threads", str(cpus),
        "--hacksize", str(HACK),
        "--ratio", str(ratio)
    ]
    subprocess.run(cmd, check=True)
    
    # Clean up parts directory immediately after merge to save disk space
    print("Cleaning up parts directory ...")
    shutil.rmtree(parts_dir)
    
    # Run rust_indexing concat
    sa_path = os.path.join(save_dir, "sa_data.sdsl")
    bwt_path = os.path.join(save_dir, "bwt_data.sdsl")
    print("Running rust_indexing concat ...")
    cmd = [
        rust_bin, "concat",
        "--data-file", os.path.abspath(ds_path),
        "--merged-dir", os.path.abspath(merged_dir),
        "--merged-file", os.path.abspath(sa_path),
        "--bwt-dir", os.path.abspath(bwt_dir),
        "--bwt-file", os.path.abspath(bwt_path),
        "--num-threads", str(cpus),
        "--ratio", str(ratio)
    ]
    subprocess.run(cmd, check=True)
    
    # Run cpp_indexing
    print("Running cpp_indexing ...")
    cmd = [cpp_bin, os.path.abspath(save_dir)]
    subprocess.run(cmd, check=True)
    
    # Clean up temporary files to save disk space
    print("Cleaning up temporary files ...")
    shutil.rmtree(merged_dir)
    shutil.rmtree(bwt_dir)
    # We also don't need the raw text_data.sdsl anymore if the index is built!
    # Wait, does InfiniGramMiniEngine need text_data.sdsl?
    # Let's check cpp_engine.h: it doesn't seem to open text_data.sdsl!
    # It opens data.fm9, data_offset, meta.fm9, meta_offset.
    # So we can delete text_data.sdsl and text_meta.sdsl!
    os.remove(ds_path)
    os.remove(mt_path)
    
    end_time = time.time()
    print(f"=== Finished {batch_name} in {end_time - start_time:.2f} seconds ===")

def main():
    repo_src = "/usr/local/google/home/lamort/Documents/fact-mem/infini-gram-mini/src"
    npy_path = "/usr/local/google/home/lamort/Documents/fact-mem/pile-tokenized/massive_tokens.npy"
    output_base_dir = "/usr/local/google/home/lamort/Documents/fact-mem/indexes"
    
    # Process in batches of 10 shards to save disk space
    # Total shards = 95
    batches = [
        (0, 10),
        (10, 20),
        (20, 30),
        (30, 40),
        (40, 50),
        (50, 60),
        (60, 70),
        (70, 80),
        (80, 90),
        (90, 95),
    ]
    
    for start, end in batches:
        process_batch(start, end, repo_src, npy_path, output_base_dir)
        
if __name__ == "__main__":
    main()
