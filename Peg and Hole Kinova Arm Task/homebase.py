#!/usr/bin/env python3
"""
Script to find and save your drop-off position
Manually jog the robot to your desired drop-off location, then run this script
"""

import collections
import collections.abc

# Fix for Python 3.10+ compatibility with old protobuf
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, "Sequence"):
    collections.Sequence = collections.abc.Sequence
if not hasattr(collections, "MutableSequence"):
    collections.MutableSequence = collections.abc.MutableSequence
if not hasattr(collections, "MutableSet"):
    collections.MutableSet = collections.abc.MutableSet

import sys
import os
import numpy as np

# Add utilities path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import utilities



from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from get_angles_pose import get_angles_pose

def main():
    # Parse connection arguments
    args = utilities.parseConnectionArguments()
    
    # Connect to robot
    print("Connecting to robot...")
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        print("✓ Connected to robot\n")
        
        # Get current position
        print("Getting current arm position...")
        angles, pose = get_angles_pose(base)
        
        print("\n" + "="*60)
        print("CURRENT POSITION")
        print("="*60)
        print("\nJoint Angles (degrees):")
        for i, angle in enumerate(angles):
            print(f"  Joint {i+1}: {angle:.4f}")

        print(f"\nCartesian Position:")
        print(f"  X: {pose.x:.4f} meters")
        print(f"  Y: {pose.y:.4f} meters")
        print(f"  Z: {pose.z:.4f} meters")
        print(f"\nOrientation:")
        print(f"  Theta X: {pose.theta_x:.2f} degrees")
        print(f"  Theta Y: {pose.theta_y:.2f} degrees")
        print(f"  Theta Z: {pose.theta_z:.2f} degrees")
        
        print("\n" + "="*60)
        print("COPY THIS INTO YOUR SCRIPT")
        print("="*60)
        print("\nDROP_OFF_POSITION = np.array([")
        print(f"    [0, 1, 0, {pose.x:.4f}],   # X position")
        print(f"    [1, 0, 0, {pose.y:.4f}],   # Y position")
        print(f"    [0, 0, -1, {pose.z:.4f}],  # Z height")
        print(f"    [0, 0, 0, 1]])")
        print("\n" + "="*60)
        
        # Save to file
        with open("dropoff_position.txt", "w") as f:
            f.write("DROP_OFF_POSITION = np.array([\n")
            f.write(f"    [0, 1, 0, {pose.x:.4f}],   # X position\n")
            f.write(f"    [1, 0, 0, {pose.y:.4f}],   # Y position\n")
            f.write(f"    [0, 0, -1, {pose.z:.4f}],  # Z height\n")
            f.write(f"    [0, 0, 0, 1]])\n")
        
        print("\n✓ Position saved to 'dropoff_position.txt'")
        print("\nYou can also copy the code above directly into your script.")

if __name__ == "__main__":
    main()