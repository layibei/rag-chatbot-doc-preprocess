from utils.lock.repositories import DistributedLockRepository
from utils.logging_util import logger


class DistributedLockHelper:
    def __init__(self, repository: DistributedLockRepository):
        self.repository = repository
        self.logger = logger

    def acquire_lock(self, lock_key: str, instance_name='localhost') -> bool:
        self.logger.info(f'Getting lock for key:{lock_key}, instance:{instance_name}')
        return self.repository.acquire_lock(lock_key, instance_name)

    def release_lock(self, lock_key: str, instance_name='localhost') -> bool:
        self.logger.info(f'Releasing lock for key:{lock_key}, instance:{instance_name}')
        return self.repository.release_lock(lock_key, instance_name)
