import site
import os
import shutil
import subprocess
import sys
from pathlib import Path

def get_package_info():
    """Resolve package name from pyproject.toml."""
    try:
        import tomllib as toml  # Python 3.11+
    except ImportError:
        try:
            import pip._vendor.tomli as toml # Fallback to pip's tomli
        except ImportError:
            return "beam-fea", "beam_fea" # Static fallback

    try:
        with open("pyproject.toml", "rb") as f:
            data = toml.load(f)
            name = data.get("project", {}).get("name", "beam-fea")
            # Usually module is name with underscore
            module = name.replace("-", "_")
            return name, module
    except Exception:
        return "beam-fea", "beam_fea"

def clean_install():
    package_name, module_name = get_package_info()
    
    print(f"=== Starting Clean Re-installation of {package_name} ===")
    
    # 1. Uninstall existing versions via pip
    print("Step 1: Uninstalling existing versions via pip...")
    for _ in range(3):  # Try a few times to be sure
        result = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", package_name], 
                               capture_output=True, text=True)
        if "not installed" in result.stderr.lower() or "not installed" in result.stdout.lower():
            break
        print("  - Detected and uninstalled a version instance.")

    # 2. Manually purge ghost metadata from site-packages
    print("\nStep 2: Scanning site-packages for orphaned metadata folders...")
    site_dirs = site.getsitepackages()
    # On some systems, user site-packages are separate
    try:
        user_site = site.getusersitepackages()
        if user_site not in site_dirs:
            site_dirs.append(user_site)
    except AttributeError:
        pass

    for sp in site_dirs:
        sp_path = Path(sp)
        if not sp_path.exists():
            continue
        
        print(f"  - Searching: {sp}")
        try:
            for entry in sp_path.iterdir():
                # Look for beam_fea*.dist-info or beam-fea*.egg-info
                name_low = entry.name.lower()
                is_beam = name_low.startswith(module_name) or name_low.startswith(package_name)
                is_meta = entry.suffix in ['.dist-info', '.egg-info']
                
                if entry.is_dir() and is_beam and is_meta:
                    print(f"    [REMOVE] Found stale metadata: {entry.name}")
                    try:
                        shutil.rmtree(entry)
                    except Exception as e:
                        print(f"    [ERROR] Could not remove {entry.name}: {e}")
        except Exception as e:
            print(f"  - Error accessing {sp}: {e}")

    # 3. Perform fresh editable install
    print("\nStep 3: Performing fresh editable installation...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        print("  - Fresh installation successful.")
    except subprocess.CalledProcessError as e:
        print(f"  - [CRITICAL ERROR] Installation failed: {e}")
        sys.exit(1)

    # 4. Final verification
    print("\nStep 4: Verifying final version...")
    try:
        # We use a subprocess to avoid issues with already-loaded modules in this process
        check_cmd = [sys.executable, "-c", "import beam_fea; print(beam_fea.__version__)"]
        ver = subprocess.check_output(check_cmd, text=True).strip()
        print(f"  - Final Reported Version: {ver}")
    except Exception as e:
        print(f"  - Verification failed: {e}")

    print("\n=== Clean Re-installation Complete ===")

if __name__ == "__main__":
    # Ensure we are in the repository root
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)
    clean_install()
