#!/usr/bin/env python3
"""
Download dataset from HuggingFace Hub
Downloads gearbox_assembly_demos from rocochallenge2025 dataset
"""
from huggingface_hub import snapshot_download
from pathlib import Path
import os

def download_gearbox_demos(
    output_dir: str = "/media/mldadmin/home/s123mdg31_14/datasets",
    repo_id: str = "rocochallenge2025/rocochallenge2025",
    folder: str = "gearbox_assembly_demos_updated",
    max_workers: int = 8
):
    """
    Download gearbox assembly demos from HuggingFace.
    
    Args:
        output_dir: Local directory to save files
        repo_id: HuggingFace repository ID
        folder: Specific folder to download
        max_workers: Number of parallel download threads
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("📥 Downloading from HuggingFace Hub")
    print("=" * 70)
    print(f"Repository: {repo_id}")
    print(f"Folder: {folder}")
    print(f"Output: {output_path.absolute()}")
    print(f"Size: ~104 GB")
    print(f"Parallel downloads: {max_workers} workers")
    print("=" * 70)
    print("\n⏳ Starting download... (this may take a while)\n")
    
    try:
        # Download specific folder from the repository
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=f"{folder}/*",  # Only download files in this folder
            local_dir=str(output_path),
            local_dir_use_symlinks=False,  # Don't use symlinks, copy actual files
            max_workers=max_workers,
            resume_download=True  # Resume if interrupted
        )
        
        print("\n" + "=" * 70)
        print("✅ Download complete!")
        print("=" * 70)
        print(f"Files saved to: {local_dir}")
        
        # Show downloaded files
        demo_dir = output_path / folder
        if demo_dir.exists():
            hdf5_files = list(demo_dir.glob("*.hdf5")) + list(demo_dir.glob("*.h5"))
            print(f"\n📊 Found {len(hdf5_files)} HDF5 files")
            if hdf5_files:
                print("\nFirst 5 files:")
                for f in sorted(hdf5_files)[:5]:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    print(f"  - {f.name} ({size_mb:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        print("\nTips:")
        print("- Check your internet connection")
        print("- Make sure you have enough disk space (~104 GB)")
        print("- You can re-run this script to resume the download")
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download gearbox assembly demos from HuggingFace")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="/media/mldadmin/home/s123mdg31_14/datasets",
        help="Output directory (default: /media/mldadmin/home/s123mdg31_14/datasets)"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=8,
        help="Number of parallel download workers (default: 8)"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="gearbox_assembly_demos_updated",
        help="Folder to download (default: gearbox_assembly_demos_updated)"
    )
    
    args = parser.parse_args()
    
    success = download_gearbox_demos(
        output_dir=args.output_dir,
        folder=args.folder,
        max_workers=args.workers
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
