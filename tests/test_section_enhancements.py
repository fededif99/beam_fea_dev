import numpy as np

from beam_fea.cross_sections import (
    rectangular, circular, i_beam, l_section, offset_section, SectionProperties
)

def test_rectangular():
    print("Testing Rectangular Section...")
    props = rectangular(100, 200) # b=100, h=200
    print(props)
    assert props.y_centroid == 0.0
    assert props.z_centroid == 0.0
    assert props.y_top == 100.0
    assert props.y_bottom == -100.0
    assert props.z_left == -50.0
    assert props.z_right == 50.0
    print("PASSED: Rectangular Section\n")

def test_l_section():
    print("Testing L-Section...")
    # L 100x100x10 (Equal angle)
    props = l_section(100, 100, 10)
    print(props)
    
    # Area = 100*10 + 90*10 = 1000 + 900 = 1900
    assert props.A == 1900.0
    
    # Centroid (x_bar, y_bar from corner)
    # y_bar = (1000*50 + 900*5) / 1900 = (50000 + 4500) / 1900 = 54500 / 1900 = 28.684
    assert np.isclose(props.y_centroid, 28.684, atol=1e-3)
    assert np.isclose(props.z_centroid, 28.684, atol=1e-3)
    
    # Extreme fibers
    # y_top = 100 - 28.684 = 71.316
    # y_bottom = -28.684
    assert np.isclose(props.y_top, 71.316, atol=1e-3)
    assert np.isclose(props.y_bottom, -28.684, atol=1e-3)
    print("PASSED: L-Section\n")

def test_offset():
    print("Testing Offset Utility...")
    # Start with a rectangle at centroid
    rect = rectangular(100, 200)
    A = rect.A
    Iy_orig = rect.Iy
    Iz_orig = rect.Iz
    
    # Offset by 50mm in Y
    offset = 50.0
    rect_off = offset_section(rect, offset_y=offset)
    
    # Parallel Axis Theorem: Iz_new = Iz_orig + A * dy^2
    # Note: in my code, Iz is about the Z axis (vertical deviation y), Iy is about Y axis (horizontal deviation z)
    # So offset_y affects Iz
    expected_Iz = Iz_orig + A * offset**2
    
    print(f"Original Iz: {Iz_orig:.2f}")
    print(f"Offset Iz:   {rect_off.Iz:.2f}")
    print(f"Expected Iz: {expected_Iz:.2f}")
    
    assert np.isclose(rect_off.Iz, expected_Iz)
    assert np.isclose(rect_off.y_centroid, -offset) # Centroid relative to NEW reference is -50
    print("PASSED: Offset Utility\n")

if __name__ == "__main__":
    try:
        test_rectangular()
        test_l_section()
        test_offset()
        print("ALL TESTS PASSED! 🎉")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
    except Exception as e:
        print(f"ERROR: {e}")
