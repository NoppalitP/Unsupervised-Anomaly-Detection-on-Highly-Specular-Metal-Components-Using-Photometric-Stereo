import unittest
import numpy as np
import torch
import cv2
import sys
import os

# Ensure the src directory is in the path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from iad_dataset_generator import PhotometricStereoSolver, AutoCropper

class TestIADPipeline(unittest.TestCase):

    def setUp(self):
        # Simple 4-light setup at 45 degree slant
        slant = np.radians(45)
        self.L = np.zeros((4, 3), dtype=np.float32)
        for i in range(4):
            az = 2.0 * np.pi * i / 4
            self.L[i, 0] = np.cos(az) * np.sin(slant)
            self.L[i, 1] = np.sin(az) * np.sin(slant)
            self.L[i, 2] = np.cos(slant)

    # =========================================================================
    # 1. PhotometricStereoSolver Tests
    # =========================================================================

    def test_ps_solver_init(self):
        solver = PhotometricStereoSolver(self.L, drop_dark=1, drop_bright=1)
        self.assertEqual(solver.n_lights, 4)
        self.assertEqual(solver.drop_dark, 1)
        self.assertEqual(solver.drop_bright, 1)

    def test_ps_solver_weight_mask(self):
        solver = PhotometricStereoSolver(self.L, drop_dark=1, drop_bright=1)
        # 1 pixel, 4 lights with intensities [10, 50, 100, 200]
        I_valid = np.array([[10, 50, 100, 200]], dtype=np.float32)
        mask = solver._build_weight_mask(I_valid)
        
        # Darkest (10) and Brightest (200) should be 0. Others should be 1.
        self.assertEqual(mask[0, 0], 0.0)  # 10
        self.assertEqual(mask[0, 1], 1.0)  # 50
        self.assertEqual(mask[0, 2], 1.0)  # 100
        self.assertEqual(mask[0, 3], 0.0)  # 200

    def test_ps_solver_gray_stack(self):
        solver = PhotometricStereoSolver(np.eye(3), drop_dark=0, drop_bright=0)
        # Create BGR image (Blue)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :, 0] = 255 # B=255
        
        gray_stack = solver._to_gray_stack([img])
        self.assertEqual(gray_stack.shape, (1, 10, 10))
        # Gray value for pure blue is approx 29 (0.114 * 255)
        self.assertTrue(28 <= gray_stack[0, 0, 0] <= 30)

    def test_ps_solver_solve_flat_plane(self):
        # If all images are identical (flat white plane), normals should point straight up (0,0,1)
        solver = PhotometricStereoSolver(self.L, drop_dark=0, drop_bright=0)
        h, w = 50, 50
        # Create 4 identical images (white square in middle)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(img, (10, 10), (40, 40), (255, 255, 255), -1)
        images = [img.copy() for _ in range(4)]
        
        result = solver.solve(images)
        
        # Center pixel should have Nx=128, Ny=128, Nz=255 (encoded 0,0,1)
        # Note: solver stores BGR as (Nz, Ny, Nx)
        # result[25, 25] -> (Nz, Ny, Nx)
        self.assertAlmostEqual(int(result[25, 25, 2]), 128, delta=1) # Nx
        self.assertAlmostEqual(int(result[25, 25, 1]), 128, delta=1) # Ny
        self.assertTrue(int(result[25, 25, 0]) > 250)  # Nz (approx 255)

    # =========================================================================
    # 2. AutoCropper Tests
    # =========================================================================

    def test_autocropper_order_points(self):
        # TL, TR, BR, BL
        pts = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.float32)
        # Shuffle them
        shuffled = pts.copy()
        np.random.shuffle(shuffled)
        
        ordered = AutoCropper._order_points(shuffled)
        self.assertTrue(np.allclose(ordered[0], [10, 10]))   # TL
        self.assertTrue(np.allclose(ordered[1], [100, 10]))  # TR
        self.assertTrue(np.allclose(ordered[2], [100, 100])) # BR
        self.assertTrue(np.allclose(ordered[3], [10, 100]))  # BL

    def test_autocropper_find_bbox(self):
        cropper = AutoCropper(output_size=100, crop_offset=0)
        # Create a white square on black background
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
        
        bbox = cropper.find_bbox([img])
        self.assertIsNotNone(bbox)
        self.assertEqual(bbox.shape, (4, 2))
        # Check if corners are approx (50,50) to (150,150)
        self.assertTrue(np.min(bbox[:, 0]) >= 48)
        self.assertTrue(np.max(bbox[:, 0]) <= 152)

    def test_autocropper_crop_and_resize(self):
        cropper = AutoCropper(output_size=100, crop_offset=0)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Draw a 100x100 square
        cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
        bbox = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.float32)
        
        cropped = cropper.crop_and_resize(img, bbox)
        self.assertEqual(cropped.shape, (100, 100, 3))
        # Cropped image should be mostly white
        self.assertTrue(np.mean(cropped) > 250)

if __name__ == '__main__':
    unittest.main()
