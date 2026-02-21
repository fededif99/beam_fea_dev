"""
Unit tests for beam_fea.materials module
"""

import pytest
from beam_fea.materials import Material, get_material, MATERIAL_LIBRARY


class TestMaterial:
    """Test Material class"""
    
    def test_valid_material_creation(self):
        """Test creating a valid material"""
        mat = Material(
            name="Test Steel",
            E=200000,
            nu=0.3,
            rho=7.85e-6
        )
        assert mat.name == "Test Steel"
        assert mat.E == 200000
        assert mat.nu == 0.3
        assert mat.rho == 7.85e-6
        # G should be calculated automatically
        assert mat.G is not None
        assert abs(mat.G - 200000 / (2 * (1 + 0.3))) < 1e-6
    
    def test_negative_youngs_modulus(self):
        """Test that negative Young's modulus raises error"""
        with pytest.raises(ValueError, match="Young's modulus E must be positive"):
            Material(name="Bad", E=-1000, nu=0.3, rho=7.85e-6)
    
    def test_zero_youngs_modulus(self):
        """Test that zero Young's modulus raises error"""
        with pytest.raises(ValueError, match="Young's modulus E must be positive"):
            Material(name="Bad", E=0, nu=0.3, rho=7.85e-6)
    
    def test_invalid_poisson_ratio_high(self):
        """Test that Poisson's ratio > 0.5 raises error"""
        with pytest.raises(ValueError, match="Poisson's ratio must be between"):
            Material(name="Bad", E=200000, nu=0.6, rho=7.85e-6)
    
    def test_invalid_poisson_ratio_low(self):
        """Test that Poisson's ratio < -1 raises error"""
        with pytest.raises(ValueError, match="Poisson's ratio must be between"):
            Material(name="Bad", E=200000, nu=-1.5, rho=7.85e-6)
    
    def test_negative_density(self):
        """Test that negative density raises error"""
        with pytest.raises(ValueError, match="Density rho must be positive"):
            Material(name="Bad", E=200000, nu=0.3, rho=-1e-6)
    
    def test_negative_shear_modulus(self):
        """Test that negative shear modulus raises error"""
        with pytest.raises(ValueError, match="Shear modulus G must be positive"):
            Material(name="Bad", E=200000, nu=0.3, rho=7.85e-6, G=-1000)
    
    def test_negative_yield_strength(self):
        """Test that negative yield strength raises error"""
        with pytest.raises(ValueError, match="Yield strength must be positive"):
            Material(name="Bad", E=200000, nu=0.3, rho=7.85e-6, yield_strength=-100)
    
    def test_ultimate_less_than_yield(self):
        """Test that ultimate < yield raises error"""
        with pytest.raises(ValueError, match="Ultimate strength .* must be >= yield strength"):
            Material(
                name="Bad", 
                E=200000, 
                nu=0.3, 
                rho=7.85e-6,
                yield_strength=500,
                ultimate_strength=400
            )
    
    def test_custom_shear_modulus(self):
        """Test providing custom shear modulus"""
        mat = Material(
            name="Custom",
            E=200000,
            nu=0.3,
            rho=7.85e-6,
            G=80000
        )
        assert mat.G == 80000


class TestGetMaterial:
    """Test get_material function"""
    
    def test_get_steel(self):
        """Test getting steel material"""
        steel = get_material('steel')
        assert steel.name == "Steel (Generic Structural)"
        assert steel.E == 200000
    
    def test_get_aluminum(self):
        """Test getting aluminum material"""
        aluminum = get_material('aluminum')
        assert aluminum.name == "Aluminum (Generic)"
    
    def test_case_insensitive(self):
        """Test that material lookup is case insensitive"""
        steel1 = get_material('STEEL')
        steel2 = get_material('steel')
        steel3 = get_material('Steel')
        assert steel1.name == steel2.name == steel3.name
    
    def test_invalid_material(self):
        """Test that invalid material name raises error"""
        with pytest.raises(ValueError, match="Material .* not found"):
            get_material('nonexistent_material')
    
    def test_all_library_materials(self):
        """Test that all materials in library can be retrieved"""
        for key in MATERIAL_LIBRARY.keys():
            mat = get_material(key)
            assert mat is not None
            assert isinstance(mat, Material)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
