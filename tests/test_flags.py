"""Tests for coordinate cleaning flags."""

import pytest
import numpy as np
from coordinatecleaner import (
    cc_val, cc_zero, cc_equ, cc_dupl, cc_round,
    cc_gbif, cc_inst, cc_cap, cc_outl, cc_iucn,
    clean_coordinates
)


class TestCcVal:
    """Test invalid coordinate detection."""
    
    def test_valid_coordinates(self):
        """Valid coordinates should pass."""
        lon = np.array([-122.4, 0, 180])
        lat = np.array([37.8, 0, -90])
        
        result = cc_val(lon, lat)
        assert np.all(result)
    
    def test_invalid_latitude(self):
        """Latitude outside [-90, 90] should fail."""
        lon = np.array([0, 0, 0])
        lat = np.array([91, -91, 200])
        
        result = cc_val(lon, lat)
        assert np.all(~result)
    
    def test_invalid_longitude(self):
        """Longitude outside [-180, 180] should fail."""
        lon = np.array([181, -181, 360])
        lat = np.array([0, 0, 0])
        
        result = cc_val(lon, lat)
        assert np.all(~result)
    
    def test_nan_values(self):
        """NaN values should fail."""
        lon = np.array([np.nan, 0])
        lat = np.array([0, np.nan])
        
        result = cc_val(lon, lat)
        assert np.all(~result)


class TestCcZero:
    """Test zero coordinate detection."""
    
    def test_not_at_zero(self):
        """Points away from (0, 0) should pass."""
        lon = np.array([-122.4, 10, 50])
        lat = np.array([37.8, 10, 50])
        
        result = cc_zero(lon, lat)
        assert np.all(result)
    
    def test_at_zero(self):
        """Points at (0, 0) should fail."""
        lon = np.array([0, 0.1, 1.0])
        lat = np.array([0, 0.1, 1.0])
        
        result = cc_zero(lon, lat, buffer=0.5)
        # First two are within buffer, third is outside
        assert not result[0]  # (0, 0) fails
        assert not result[1]  # (0.1, 0.1) within buffer
        assert result[2]      # (1.0, 1.0) outside buffer


class TestCcEqu:
    """Test equal lat/lon detection."""
    
    def test_not_equal(self):
        """Points with different lat/lon should pass."""
        lon = np.array([-122.4, 10, 50])
        lat = np.array([37.8, 20, 60])
        
        result = cc_equ(lon, lat)
        assert np.all(result)
    
    def test_equal(self):
        """Points with lat == lon should fail."""
        lon = np.array([37.8, 45.0, 0.0])
        lat = np.array([37.8, 45.0, 0.0])
        
        result = cc_equ(lon, lat)
        assert np.all(~result)


class TestCcDupl:
    """Test duplicate detection."""
    
    def test_no_duplicates(self):
        """Unique points should all pass."""
        lon = np.array([1, 2, 3, 4])
        lat = np.array([1, 2, 3, 4])
        
        result = cc_dupl(lon, lat)
        assert np.all(result)
    
    def test_with_duplicates(self):
        """Duplicate points should fail (except first)."""
        lon = np.array([1, 2, 1, 3, 2])  # 1 and 2 repeated
        lat = np.array([1, 2, 1, 3, 2])
        
        result = cc_dupl(lon, lat)
        assert result[0] and result[1] and result[3]  # First occurrences pass
        assert not result[2] and not result[4]  # Duplicates fail


class TestCcRound:
    """Test rounded coordinate detection."""
    
    def test_precise_coordinates(self):
        """Precise coordinates should pass."""
        lon = np.array([-122.4194, 10.12345])
        lat = np.array([37.7749, 20.54321])
        
        result = cc_round(lon, lat, decimals=2)
        assert np.all(result)
    
    def test_rounded_coordinates(self):
        """Heavily rounded coordinates should fail."""
        lon = np.array([15.0, 45.0])
        lat = np.array([47.0, 10.0])
        
        result = cc_round(lon, lat, decimals=2)
        assert np.all(~result)


class TestCcGbif:
    """Test GBIF HQ detection."""
    
    def test_not_at_gbif(self):
        """Points away from GBIF HQ should pass."""
        lon = np.array([-122.4, 0, 50])
        lat = np.array([37.8, 0, 50])
        
        result = cc_gbif(lon, lat)
        assert np.all(result)
    
    def test_at_gbif(self):
        """Point at GBIF HQ should fail."""
        lon = np.array([-0.079])
        lat = np.array([51.52])
        
        result = cc_gbif(lon, lat)
        assert not result[0]


class TestCcInst:
    """Test institution detection."""
    
    def test_not_at_museum(self):
        """Points away from museums should pass."""
        lon = np.array([-122.4, 0, 50])
        lat = np.array([37.8, 0, 50])
        
        result = cc_inst(lon, lat)
        assert np.all(result)
    
    def test_at_museum(self):
        """Point at Natural History Museum London should fail."""
        lon = np.array([-0.1766])
        lat = np.array([51.4966])
        
        result = cc_inst(lon, lat, buffer=0.001)
        assert not result[0]


class TestCcCap:
    """Test capital detection."""
    
    def test_not_at_capital(self):
        """Points away from capitals should pass."""
        lon = np.array([-122.4, 0, 50])
        lat = np.array([37.8, 0, 50])
        
        result = cc_cap(lon, lat)
        assert np.all(result)
    
    def test_at_capital(self):
        """Point at Washington DC should fail."""
        lon = np.array([-77.0369])
        lat = np.array([38.9072])
        
        result = cc_cap(lon, lat)
        assert not result[0]


class TestCleanCoordinates:
    """Test combined cleaning function."""
    
    def test_all_clean(self):
        """Clean coordinates should pass all tests."""
        lon = np.array([-122.4194, 10.12345, 50.54321])
        lat = np.array([37.7749, 20.54321, 30.12345])
        
        result = clean_coordinates(lon, lat)
        assert np.all(result)
    
    def test_with_problems(self):
        """Mixed coordinates should be filtered."""
        lon = np.array([-122.4194, 0, -0.079])  # Valid, at zero, at GBIF
        lat = np.array([37.7749, 0, 51.52])
        
        result = clean_coordinates(lon, lat)
        assert result[0]  # Valid
        assert not result[1]  # At zero
        assert not result[2]  # At GBIF HQ
    
    def test_return_details(self):
        """Should return individual test results."""
        lon = np.array([0, -122.4])
        lat = np.array([0, 37.8])
        
        valid, details = clean_coordinates(lon, lat, return_details=True)
        
        assert 'val' in details
        assert 'zero' in details
        assert len(details) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestCcOutl:
    """Test geographic outlier detection."""
    
    def test_no_outliers(self):
        """Clustered points should all pass."""
        np.random.seed(42)
        # Points clustered around (0, 0)
        lon = np.random.normal(0, 1, 20)
        lat = np.random.normal(0, 1, 20)
        
        result = cc_outl(lon, lat)
        # Most should pass
        assert np.sum(result) >= 18
    
    def test_with_outlier(self):
        """Distant point should be flagged."""
        # Cluster near (0, 0)
        lon = np.array([0, 0.1, -0.1, 0.05, -0.05, 0.1, -0.1, 100])  # Last is outlier
        lat = np.array([0, 0.1, -0.1, 0.05, -0.05, 0.1, -0.1, 100])
        
        result = cc_outl(lon, lat, threshold=3)
        
        # Outlier should fail
        assert not result[-1]
        # Others should pass
        assert np.all(result[:-1])
    
    def test_min_records(self):
        """Too few records should pass all."""
        lon = np.array([0, 100])
        lat = np.array([0, 100])
        
        result = cc_outl(lon, lat, min_records=7)
        assert np.all(result)


class TestCcIucn:
    """Test range polygon checking."""
    
    def test_no_polygon(self):
        """No polygon should pass all."""
        lon = np.array([0, 1, 2])
        lat = np.array([0, 1, 2])
        
        from coordinatecleaner import cc_iucn
        result = cc_iucn(lon, lat)
        assert np.all(result)
    
    def test_inside_polygon(self):
        """Points inside polygon should pass."""
        # Square polygon from (0,0) to (10,10)
        polygon = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        
        lon = np.array([5, 5, 5])
        lat = np.array([5, 2, 8])
        
        from coordinatecleaner import cc_iucn
        result = cc_iucn(lon, lat, range_polygon=polygon)
        assert np.all(result)
    
    def test_outside_polygon(self):
        """Points outside polygon should fail."""
        polygon = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        
        lon = np.array([15, -5])  # Outside
        lat = np.array([5, 5])
        
        from coordinatecleaner import cc_iucn
        result = cc_iucn(lon, lat, range_polygon=polygon)
        assert np.all(~result)


class TestCcSea:
    """Test land/sea coordinate checking."""
    
    def test_without_mask_warns(self):
        """Without land mask, should warn and return all valid."""
        import warnings
        lon = np.array([-122.4, -123.0])
        lat = np.array([48.5, 49.0])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from coordinatecleaner import cc_sea
            result = cc_sea(lon, lat)
            
            # Should warn about missing land mask
            assert len(w) == 1
            assert "land_mask" in str(w[0].message)
            
            # Should return all valid (can't verify)
            assert np.all(result)
    
    def test_with_land_mask(self):
        """With land mask, should correctly identify land/sea."""
        # Create simple land mask: land in center
        land_mask = np.zeros((10, 10), dtype=bool)
        land_mask[3:7, 3:7] = True  # Land in center
        
        extent = (0, 10, 0, 10)  # lon_min, lon_max, lat_min, lat_max
        
        # Point on land (center)
        lon_land = np.array([5.0])
        lat_land = np.array([5.0])
        
        # Point in water (corner)
        lon_sea = np.array([1.0])
        lat_sea = np.array([1.0])
        
        from coordinatecleaner import cc_sea
        
        # Land point should return False (not in sea)
        result_land = cc_sea(lon_land, lat_land, land_mask=land_mask, land_mask_extent=extent)
        assert not result_land[0]
        
        # Sea point should return True (in sea)
        result_sea = cc_sea(lon_sea, lat_sea, land_mask=land_mask, land_mask_extent=extent)
        assert result_sea[0]
