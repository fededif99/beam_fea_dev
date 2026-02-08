"""
Unit tests for beam_fea.cross_sections module
"""

import pytest
import numpy as np
from beam_fea.cross_sections import (
    RectangularSection, CircularSection, HollowCircularSection,
    IBeamSection, BoxSection, rectangular, circular
)


class TestRectangularSection:
    """Test RectangularSection class"""
    
    def test_valid_creation(self):
        """Test creating valid rectangular section"""
        section = RectangularSection(width=50, height=100)
        assert section.width == 50
        assert section.height == 100
    
    def test_properties_calculation(self):
        """Test that properties are calculated correctly"""
        section = RectangularSection(width=50, height=100)
        props = section.properties()
        
        # Area = b * h
        assert abs(props.A - 5000) < 1e-6
        
        # Iy = b * h^3 / 12
        expected_Iy = (50 * 100**3) / 12
        assert abs(props.Iy - expected_Iy) < 1e-6
        
        # Iz = h * b^3 / 12
        expected_Iz = (100 * 50**3) / 12
        assert abs(props.Iz - expected_Iz) < 1e-6
    
    def test_negative_width(self):
        """Test that negative width raises error"""
        with pytest.raises(ValueError, match="Width must be positive"):
            RectangularSection(width=-50, height=100)
    
    def test_zero_height(self):
        """Test that zero height raises error"""
        with pytest.raises(ValueError, match="Height must be positive"):
            RectangularSection(width=50, height=0)


class TestCircularSection:
    """Test CircularSection class"""
    
    def test_valid_creation(self):
        """Test creating valid circular section"""
        section = CircularSection(diameter=50)
        assert section.diameter == 50
    
    def test_properties_calculation(self):
        """Test that properties are calculated correctly"""
        section = CircularSection(diameter=50)
        props = section.properties()
        
        r = 25
        # Area = π * r^2
        expected_A = np.pi * r**2
        assert abs(props.A - expected_A) < 1e-6
        
        # I = π * d^4 / 64
        expected_I = (np.pi * 50**4) / 64
        assert abs(props.Iy - expected_I) < 1e-6
        assert abs(props.Iz - expected_I) < 1e-6
    
    def test_negative_diameter(self):
        """Test that negative diameter raises error"""
        with pytest.raises(ValueError, match="Diameter must be positive"):
            CircularSection(diameter=-50)


class TestHollowCircularSection:
    """Test HollowCircularSection class"""
    
    def test_valid_creation(self):
        """Test creating valid hollow circular section"""
        section = HollowCircularSection(outer_diameter=60, thickness=5)
        assert section.outer_diameter == 60
        assert section.thickness == 5
        assert section.inner_diameter == 50
    
    def test_properties_calculation(self):
        """Test that properties are calculated correctly"""
        section = HollowCircularSection(outer_diameter=60, thickness=5)
        props = section.properties()
        
        ro = 30
        ri = 25
        expected_A = np.pi * (ro**2 - ri**2)
        assert abs(props.A - expected_A) < 1e-6
    
    def test_thickness_too_large(self):
        """Test that thickness >= radius raises error"""
        with pytest.raises(ValueError, match="Thickness .* must be less than radius"):
            HollowCircularSection(outer_diameter=60, thickness=35)
    
    def test_negative_thickness(self):
        """Test that negative thickness raises error"""
        with pytest.raises(ValueError, match="Thickness must be positive"):
            HollowCircularSection(outer_diameter=60, thickness=-5)


class TestIBeamSection:
    """Test IBeamSection class"""
    
    def test_valid_creation(self):
        """Test creating valid I-beam section"""
        section = IBeamSection(
            flange_width=100,
            total_height=200,
            web_thickness=8,
            flange_thickness=12
        )
        assert section.flange_width == 100
        assert section.total_height == 200
    
    def test_properties_calculation(self):
        """Test that properties are calculated"""
        section = IBeamSection(
            flange_width=100,
            total_height=200,
            web_thickness=8,
            flange_thickness=12
        )
        props = section.properties()
        
        # Just check that values are positive and reasonable
        assert props.A > 0
        assert props.Iy > 0
        assert props.Iz > 0
    
    def test_flanges_too_thick(self):
        """Test that total flange thickness >= height raises error"""
        with pytest.raises(ValueError, match="Total flange thickness .* must be less than total height"):
            IBeamSection(
                flange_width=100,
                total_height=200,
                web_thickness=8,
                flange_thickness=105  # 2*105 > 200
            )
    
    def test_web_thicker_than_flange(self):
        """Test that web thickness > flange width raises error"""
        with pytest.raises(ValueError, match="Web thickness .* should not exceed flange width"):
            IBeamSection(
                flange_width=100,
                total_height=200,
                web_thickness=150,
                flange_thickness=12
            )


class TestBoxSection:
    """Test BoxSection class"""
    
    def test_valid_creation(self):
        """Test creating valid box section"""
        section = BoxSection(width=80, height=120, thickness=6)
        assert section.width == 80
        assert section.height == 120
        assert section.thickness == 6
    
    def test_thickness_too_large_width(self):
        """Test that thickness >= width/2 raises error"""
        with pytest.raises(ValueError, match="Thickness .* must be less than half the width"):
            BoxSection(width=80, height=120, thickness=45)
    
    def test_thickness_too_large_height(self):
        """Test that thickness >= height/2 raises error"""
        with pytest.raises(ValueError, match="Thickness .* must be less than half"):
            BoxSection(width=200, height=80, thickness=45)  # 45 > 80/2


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_rectangular_function(self):
        """Test rectangular() convenience function"""
        props = rectangular(50, 100)
        assert props.A == 5000
    
    def test_circular_function(self):
        """Test circular() convenience function"""
        props = circular(50)
        r = 25
        expected_A = np.pi * r**2
        assert abs(props.A - expected_A) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
