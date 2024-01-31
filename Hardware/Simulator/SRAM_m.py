
import math
class SRAM:
    def __init__(self,sram_size=100*8,bandwidth=76.8 * 1024 * 1024 * 1024 * 8 ):
        # SRAM
        # FIXME: too small
        # self.max_Q = 2*1024*8 # 53KB
        # self.max_K = 2*1024*8 # 53KB
        # self.max_V = 2*1024*8 # 53KB
        # self.max_index = 2*1024*8 # 53KB
        # self.max_output = 2*1024*8 # 53KB
        self.decoder_capacity = sram_size * 8
        self.encoder_capacity = sram_size* 8
        self.Q_capacity = sram_size* 8
        self.K_capacity = sram_size* 8
        self.V_capacity = sram_size* 8
        self.index_capacity = sram_size* 8
        self.output_capacity = sram_size* 8
        self.weight_capacity = sram_size* 8
        self.max_Q = sram_size* 8 # 53KB
        self.max_K = sram_size * 8# 53KB
        self.max_V = sram_size* 8 # 53KB
        self.max_index = sram_size * 8# 20KB
        self.max_output = sram_size* 8# 108KB

        # HBM to SRAM
        self.bandwidth = bandwidth# 76.8GB/s
        self.clock_frequency = 500 * 1e6 # 500MHz

    
    def preload_decoder(self, nums=0, bits=32, bandwidth_ratio=1):
        if nums * bits > self.decoder_capacity:
            self.decoder_capacity = self.max_Q - (nums*bits - self.decoder_capacity)
            latency = (nums * bits + self.max_Q) / (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)
            status = 'clear'
        else:
            self.decoder_capacity = self.decoder_capacity - nums*bits
            latency = nums * bits / (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)
            status = 'none'
        return cycle , status

    def preload_encoder(self, nums=0, bits=32, bandwidth_ratio=1):
        if nums * bits > self.encoder_capacity:
            self.encoder_capacity = self.max_Q - (nums*bits - self.encoder_capacity)
            latency = (nums * bits + self.max_Q)/ (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)
            status = 'clear'
        else:
            self.encoder_capacity = self.encoder_capacity - nums*bits 
            latency = nums * bits / (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)
            status = 'none'
        return cycle , status
    
    def preload_Q(self, nums=0, bits=32, bandwidth_ratio=1):
        if nums * bits > self.Q_capacity:
            self.Q_capacity = self.max_Q - (nums*bits - self.Q_capacity)
            latency = (nums * bits + self.max_Q)/ (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)
            status = 'clear'
        else:
            self.Q_capacity = self.Q_capacity - nums*bits 
            latency = nums * bits / (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)
            status = 'none'
        return cycle , status
    
    def data_cycle(self, num, bit=8, bandwidth_ratio=1):
        latency = (num*bit) / (self.bandwidth * bandwidth_ratio)
        cycle = math.ceil(latency * self.clock_frequency)
        return cycle

    def preload_K(self, nums=0, bits=32, bandwidth_ratio=1):
        if nums * bits > self.K_capacity:
            self.K_capacity = self.max_K - (nums*bits - self.K_capacity)
            latency = (nums * bits + self.max_Q) / (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)
            status = 'clear'
        else:
            self.K_capacity = self.K_capacity - nums*bits 
            latency = nums * bits / (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)
            status = 'none'
        return cycle , status

    def preload_V(self, nums=0, bits=32, bandwidth_ratio=1):
        if nums * bits > self.V_capacity:
            self.V_capacity = self.max_V - (nums*bits - self.V_capacity)
            latency = (nums * bits + self.max_Q) / (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)
            status = 'clear'
        else:
            self.V_capacity = self.V_capacity - nums*bits 
            latency = nums * bits / (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)
            status = 'none'
        return cycle , status

    def preload_index(self, nums=0, bits=32, bandwidth_ratio=1):
        if nums * bits > self.index_capacity:
            self.index_capacity = self.max_index - (nums*bits - self.index_capacity)
            latency = (nums * bits + self.max_Q)/ (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)
            status = 'clear'
        else:
            self.index_capacity = self.index_capacity - nums*bits 
            latency = nums * bits / (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)
            status = 'none'
        return cycle , status

    def store_out(self, nums=0, bits=32, bandwidth_ratio=1):
        if nums * bits > self.output_capacity:
            self.output_capacity = self.max_output - (nums*bits - self.output_capacity)
            latency = (nums * bits + self.max_Q) / (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)
            status = 'clear'
        else:
            self.output_capacity = self.output_capacity - nums*bits 
            latency = nums * bits / (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)
            status = 'none'
        return cycle , status

    
    def preload_weight(self, nums=0, bits=32, bandwidth_ratio=1):
        if nums * bits > self.weight_capacity:
            self.weight_capacity = self.max_output - (nums*bits - self.weight_capacity)
            latency = (nums * bits + self.max_Q) / (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)
            status = 'clear'
        else:
            self.weight_capacity = self.weight_capacity - nums*bits 
            latency = nums * bits / (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)
            status = 'none'
        return cycle , status