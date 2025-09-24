import os
import time
import random
import struct

def print_info():
    """Print system information"""
    print(f"Python PID: {os.getpid()}")
    print(f"UID: {os.getuid()}")
    print(f"Testing PCIe configuration space access...\n")

def find_pci_devices():
    """Find available PCI devices in the system"""
    devices = []
    try:
        for device in os.listdir('/sys/bus/pci/devices'):
            devices.append(device)
        return devices
    except FileNotFoundError:
        print("PCI sysfs not found. Are you running with proper permissions?")
        return []

def read_pcie_config(device, offset, size=1):
    """
    Read PCIe configuration space
    
    Args:
        device: PCI device address (e.g., "0000:00:00.0")
        offset: Configuration space offset
        size: Size to read (1, 2, or 4 bytes)
    
    Returns:
        Value read from configuration space
    """
    try:
        path = f"/sys/bus/pci/devices/{device}/config"
        with open(path, "rb") as f:
            f.seek(offset)
            if size == 1:
                data = f.read(1)
                return struct.unpack('B', data)[0]
            elif size == 2:
                data = f.read(2)
                return struct.unpack('<H', data)[0]  # Little-endian
            elif size == 4:
                data = f.read(4)
                return struct.unpack('<I', data)[0]  # Little-endian
    except Exception as e:
        print(f"Error reading config space: {e}")
        return None

def write_pcie_config(device, offset, value, size=1):
    """
    Write to PCIe configuration space (requires root privileges)
    
    Args:
        device: PCI device address (e.g., "0000:00:00.0")
        offset: Configuration space offset
        value: Value to write
        size: Size to write (1, 2, or 4 bytes)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        path = f"/sys/bus/pci/devices/{device}/config"
        with open(path, "r+b") as f:
            f.seek(offset)
            if size == 1:
                f.write(struct.pack('B', value))
            elif size == 2:
                f.write(struct.pack('<H', value))  # Little-endian
            elif size == 4:
                f.write(struct.pack('<I', value))  # Little-endian
            return True
    except Exception as e:
        print(f"Error writing config space: {e}")
        return False

def safe_regions(device):
    """
    Returns "safe" regions for read/write testing that won't disrupt system operation
    Typically vendor/device ID (read-only) and some device-specific regions
    
    Args:
        device: PCI device address
    
    Returns:
        Dictionary of safe regions and access types
    """
    # Read vendor and device ID to identify the device
    vendor_id = read_pcie_config(device, 0x00, 2)
    device_id = read_pcie_config(device, 0x02, 2)
    
    print(f"Device {device}: Vendor ID 0x{vendor_id:04x}, Device ID 0x{device_id:04x}")
    
    # Default safe regions
    regions = {
        # Read-only regions
        "read_only": [
            (0x00, 2),  # Vendor ID
            (0x02, 2),  # Device ID
            (0x08, 1),  # Revision ID
            (0x0E, 1),  # Header Type
        ],
        # Read-write regions (be careful, these are generally safe but could vary)
        "read_write": []
    }
    
    # Only attempt writes on devices with known safe regions
    # These are just examples - real safe regions depend on the specific device
    if (vendor_id == 0x8086):  # Intel
        regions["read_write"] = [(0x4C, 1)]  # Often a scratch register on Intel devices
    
    return regions

def main():
    print_info()
    print("Waiting 5 seconds to give you time to attach tracer...")
    time.sleep(5)
    
    # Find PCI devices
    devices = find_pci_devices()
    if not devices:
        print("No PCI devices found.")
        return
    
    print(f"Found {len(devices)} PCI devices")
    
    # Select some devices for testing (limit to 3 to avoid overwhelming output)
    test_devices = devices[:3]
    print(f"Testing with devices: {test_devices}")
    
    # Perform multiple rounds of reads
    for round_num in range(1, 4):
        print(f"\nRound {round_num} - Testing PCIe config space reads")
        
        for device in test_devices:
            safe = safe_regions(device)
            
            # Read tests
            print(f"\nPerforming reads on {device}:")
            
            # Test byte reads
            for offset, size in safe["read_only"]:
                if size == 1:
                    value = read_pcie_config(device, offset, 1)
                    print(f"  Byte read at offset 0x{offset:02x}: 0x{value:02x}")
                    time.sleep(0.05)
            
            # Test word reads
            for offset, size in safe["read_only"]:
                if size == 2 and offset % 2 == 0:  # Ensure aligned
                    value = read_pcie_config(device, offset, 2)
                    print(f"  Word read at offset 0x{offset:02x}: 0x{value:04x}")
                    time.sleep(0.05)
            
            # Test dword reads
            for offset in range(0, 0x40, 4):  # Read some configuration dwords
                value = read_pcie_config(device, offset, 4)
                print(f"  Dword read at offset 0x{offset:02x}: 0x{value:08x}")
                time.sleep(0.05)
    
    # Write tests (only if we're root and have safe regions)
    if os.getuid() == 0:  # Only root can write
        print("\nTesting PCIe config space writes (requires root)")
        
        for device in test_devices:
            safe = safe_regions(device)
            if not safe["read_write"]:
                print(f"No safe write regions identified for {device}, skipping writes")
                continue
                
            print(f"\nPerforming writes on {device}:")
            
            for offset, size in safe["read_write"]:
                # Read original value
                original = read_pcie_config(device, offset, size)
                print(f"  Original value at offset 0x{offset:02x}: 0x{original:x}")
                
                # Write same value back (should be safe)
                if write_pcie_config(device, offset, original, size):
                    print(f"  Wrote value 0x{original:x} to offset 0x{offset:02x}")
                
                # Verify
                new_value = read_pcie_config(device, offset, size)
                print(f"  Verification read: 0x{new_value:x}")
                
                time.sleep(0.1)
    else:
        print("\nSkipping write tests - requires root privileges")
    
    print("\nPCIe testing complete!")

if __name__ == "__main__":
    main()