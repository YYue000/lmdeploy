from ...messages import SchedulerSequence

class CPUBlockManager():
    
    def can_allocate(self, msg: SchedulerSequence, prealloc_size: int = 0):
        return True
    
    def free(self, msg: SchedulerSequence):
        return
    
    def get_block_table(self, msg: SchedulerSequence):
        return None
    
    def allocate(self, data: SchedulerSequence, prealloc_size: int = 0):
        return