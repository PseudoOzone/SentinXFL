"""
Multi-organization federation manager for SentinXFL v2.0

Manages multiple participating organizations in federated learning:
- Organization registration and tracking
- Heartbeat mechanism for health monitoring
- Asynchronous aggregation (K-of-N participants)
- Model version management
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class Organization:
    """
    Represents a participating organization in federated learning.
    
    Attributes:
        org_id: Unique organization identifier
        region: Geographic region (e.g., 'US-East', 'EU-West')
        public_key: Organization's public encryption key
        last_heartbeat: Timestamp of last heartbeat signal
        model_version: Current model version number
    """
    
    def __init__(self, org_id: str, region: str):
        """
        Initialize organization.
        
        Args:
            org_id: Unique organization identifier
            region: Geographic region
        """
        self.org_id = org_id
        self.region = region
        self.public_key: Optional[str] = None
        self.last_heartbeat: float = time.time()
        self.model_version: int = 0
        self.is_active: bool = True
        
    def heartbeat(self) -> None:
        """Update last heartbeat timestamp."""
        self.last_heartbeat = time.time()
        
    def __repr__(self) -> str:
        return f"Organization(org_id={self.org_id}, region={self.region}, version={self.model_version})"


class OrganizationManager:
    """
    Manages multiple organizations and async federated learning.
    
    Supports:
    - Organization registration and tracking
    - Asynchronous aggregation (K-of-N protocol)
    - Heartbeat monitoring
    - Model version control
    """
    
    def __init__(self):
        """Initialize organization manager."""
        self.organizations: Dict[str, Organization] = {}
        self.rounds: int = 0
        logger.info("OrganizationManager initialized")
    
    def register_org(self, org_id: str, region: str) -> Organization:
        """
        Register a new organization.
        
        Args:
            org_id: Unique organization ID
            region: Geographic region
            
        Returns:
            Organization: The newly registered organization
            
        Raises:
            ValueError: If org_id already exists
        """
        if org_id in self.organizations:
            raise ValueError(f"Organization {org_id} already registered")
            
        org = Organization(org_id, region)
        self.organizations[org_id] = org
        logger.info(f"Registered organization: {org_id} in {region}")
        return org
    
    def unregister_org(self, org_id: str) -> bool:
        """
        Unregister an organization.
        
        Args:
            org_id: Organization ID to unregister
            
        Returns:
            bool: True if successful, False if org not found
        """
        if org_id in self.organizations:
            del self.organizations[org_id]
            logger.info(f"Unregistered organization: {org_id}")
            return True
        return False
    
    def get_active_orgs(self, timeout: int = 30) -> List[Organization]:
        """
        Get currently active organizations.
        
        An organization is considered active if it has sent a heartbeat
        within the specified timeout period.
        
        Args:
            timeout: Max seconds since last heartbeat (default: 30)
            
        Returns:
            List[Organization]: List of active organizations
        """
        current_time = time.time()
        active = [org for org in self.organizations.values()
                  if (current_time - org.last_heartbeat < timeout) and org.is_active]
        return active
    
    def get_inactive_orgs(self, timeout: int = 30) -> List[Organization]:
        """
        Get currently inactive organizations.
        
        An organization is considered inactive if it has not sent a heartbeat
        within the specified timeout period.
        
        Args:
            timeout: Max seconds since last heartbeat (default: 30)
            
        Returns:
            List[Organization]: List of inactive organizations
        """
        current_time = time.time()
        inactive = [org for org in self.organizations.values()
                    if (current_time - org.last_heartbeat >= timeout) or not org.is_active]
        return inactive
    
    def heartbeat(self, org_id: str) -> bool:
        """
        Record heartbeat from organization.
        
        Args:
            org_id: Organization ID
            
        Returns:
            bool: True if successful, False if org not found
        """
        if org_id in self.organizations:
            self.organizations[org_id].heartbeat()
            return True
        return False
    
    def async_aggregate(self, 
                       weights_dict: Dict[str, np.ndarray],
                       min_orgs: int = 5,
                       timeout: int = 10) -> Tuple[Optional[np.ndarray], int]:
        """
        Perform asynchronous FedAvg aggregation.
        
        Waits for minimum number of organizations to submit weights,
        then aggregates. Does not wait for all organizations.
        
        Args:
            weights_dict: {org_id: weights_array} dictionary
            min_orgs: Minimum organizations needed to aggregate (default: 5)
            timeout: Timeout in seconds (default: 10)
            
        Returns:
            Tuple[aggregated_weights, num_orgs_used]: Aggregated weights and count
        """
        start_time = time.time()
        received_orgs = len(weights_dict)
        
        # Check if we have minimum participants
        if received_orgs < min_orgs:
            logger.warning(f"Insufficient organizations: {received_orgs} < {min_orgs}")
            return None, 0
        
        # Simple FedAvg: average all weights
        try:
            weight_arrays = list(weights_dict.values())
            aggregated = np.mean(weight_arrays, axis=0)
            logger.info(f"Aggregated {received_orgs} organizations (min: {min_orgs})")
            return aggregated, received_orgs
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return None, 0
    
    def get_organization(self, org_id: str) -> Optional[Organization]:
        """
        Get organization by ID.
        
        Args:
            org_id: Organization ID
            
        Returns:
            Organization: The organization, or None if not found
        """
        return self.organizations.get(org_id)
    
    def get_all_organizations(self) -> List[Organization]:
        """
        Get all registered organizations.
        
        Returns:
            List[Organization]: All organizations
        """
        return list(self.organizations.values())
    
    def get_org_count(self) -> int:
        """Get total number of registered organizations."""
        return len(self.organizations)
    
    def get_active_count(self, timeout: int = 30) -> int:
        """Get count of active organizations."""
        return len(self.get_active_orgs(timeout))
    
    def increment_round(self) -> None:
        """Increment round counter."""
        self.rounds += 1
        logger.debug(f"Round {self.rounds} completed")
    
    def __repr__(self) -> str:
        active = self.get_active_count()
        total = self.get_org_count()
        return f"OrganizationManager(active={active}/{total}, rounds={self.rounds})"
