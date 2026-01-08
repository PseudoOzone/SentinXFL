"""
Tests for multi-organization federation manager

Tests the OrganizationManager and Organization classes:
- Registration and unregistration
- Heartbeat mechanism
- Active/inactive organization tracking
- Async aggregation
"""

import pytest
import time
import numpy as np
from src.federation.multi_org_manager import OrganizationManager, Organization


class TestOrganization:
    """Test Organization class."""
    
    def test_organization_creation(self):
        """Test creating an organization."""
        org = Organization("org_1", "US-East")
        assert org.org_id == "org_1"
        assert org.region == "US-East"
        assert org.is_active is True
    
    def test_organization_heartbeat(self):
        """Test organization heartbeat."""
        org = Organization("org_1", "US-East")
        old_time = org.last_heartbeat
        time.sleep(0.1)
        org.heartbeat()
        assert org.last_heartbeat > old_time


class TestOrganizationManager:
    """Test OrganizationManager class."""
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = OrganizationManager()
        assert manager.get_org_count() == 0
        assert manager.rounds == 0
    
    def test_register_organization(self):
        """Test registering organizations."""
        manager = OrganizationManager()
        
        for i in range(5):
            manager.register_org(f"org_{i}", f"region_{i % 2}")
        
        assert manager.get_org_count() == 5
    
    def test_register_duplicate_org(self):
        """Test registering duplicate organization."""
        manager = OrganizationManager()
        manager.register_org("org_1", "region_1")
        
        with pytest.raises(ValueError):
            manager.register_org("org_1", "region_2")
    
    def test_unregister_organization(self):
        """Test unregistering organization."""
        manager = OrganizationManager()
        manager.register_org("org_1", "region_1")
        
        assert manager.unregister_org("org_1") is True
        assert manager.get_org_count() == 0
    
    def test_active_organizations(self):
        """Test getting active organizations."""
        manager = OrganizationManager()
        
        # Register 5 organizations
        for i in range(5):
            manager.register_org(f"org_{i}", f"region_{i}")
        
        # All should be active initially
        active = manager.get_active_orgs(timeout=30)
        assert len(active) == 5
    
    def test_inactive_organizations(self):
        """Test detecting inactive organizations."""
        manager = OrganizationManager()
        
        # Register 2 organizations
        org1 = manager.register_org("org_1", "region_1")
        org2 = manager.register_org("org_2", "region_2")
        
        # Make org2 inactive by setting old heartbeat
        org2.last_heartbeat = time.time() - 60  # 60 seconds ago
        
        # org1 should be active, org2 inactive
        active = manager.get_active_orgs(timeout=30)
        assert len(active) == 1
        assert active[0].org_id == "org_1"
    
    def test_heartbeat(self):
        """Test heartbeat mechanism."""
        manager = OrganizationManager()
        manager.register_org("org_1", "region_1")
        
        # Heartbeat should succeed
        assert manager.heartbeat("org_1") is True
        
        # Heartbeat for non-existent org should fail
        assert manager.heartbeat("org_999") is False
    
    def test_async_aggregation(self):
        """Test asynchronous aggregation."""
        manager = OrganizationManager()
        
        # Create 10 organizations with dummy weights
        weights_dict = {}
        for i in range(10):
            weights = np.ones(768) * (i + 1)  # Weights from 1 to 10
            weights_dict[f"org_{i}"] = weights
        
        # Aggregate with min_orgs=5
        aggregated, count = manager.async_aggregate(weights_dict, min_orgs=5)
        
        assert aggregated is not None
        assert count == 10
        # Average should be around 5.5 (mean of 1 to 10)
        assert np.mean(aggregated) > 5
    
    def test_async_aggregation_insufficient_orgs(self):
        """Test aggregation with insufficient organizations."""
        manager = OrganizationManager()
        
        weights_dict = {"org_1": np.ones(768)}
        
        # Try to aggregate with min_orgs=5, but only 1 available
        aggregated, count = manager.async_aggregate(weights_dict, min_orgs=5)
        
        assert aggregated is None
        assert count == 0
    
    def test_get_organization(self):
        """Test getting organization by ID."""
        manager = OrganizationManager()
        manager.register_org("org_1", "region_1")
        
        org = manager.get_organization("org_1")
        assert org is not None
        assert org.org_id == "org_1"
        
        org = manager.get_organization("org_999")
        assert org is None
    
    def test_increment_round(self):
        """Test round counter."""
        manager = OrganizationManager()
        assert manager.rounds == 0
        
        manager.increment_round()
        assert manager.rounds == 1
        
        manager.increment_round()
        assert manager.rounds == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
