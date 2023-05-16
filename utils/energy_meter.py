import threading
import time
from queue import Queue

class EnergyMeter(threading.Thread):
	def __init__(self, device, sleep_time, mode, out_dir, filename, discarded):
		self.device = device
		self.sleep_time = float(sleep_time / 1000.0)
		self.mode = mode
		self.out_dir = out_dir
		self.filename = filename
		self.queue = Queue()
		self.max_energy = 0.0
		self.min_energy = 10000.0
		self.total_energy = 0.0
		self.time = 0.0
		self.steps = 0
		self.sensors = dict()
		self.discarded = discarded
		if self.device == 'nano':
			# Whole board
			self.sensors['POM_5V_IN'] = open('/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_current0_input', 'r')
			# GPU
			self.sensors['POM_5V_GPU'] = open('/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_current1_input', 'r')
			# CPU
			self.sensors['POM_5V_CPU'] = open('/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_current2_input', 'r')

			if self.mode == 'GPU':
				self.mode = 'POM_5V_GPU'
			else:
				self.mode = 'POM_5V_CPU'
		elif self.device == 'xavier':
			# GPU
			self.sensors['GPU'] = open('/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_current0_input', 'r')
			# CPU
			self.sensors['CPU'] = open('/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_current1_input', 'r')
			# SoC - GPU - CPU (rest of SoC components)
			self.sensors['SOC'] = open('/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_current2_input', 'r')
			# Computer Vision modules + Deep Learning Accelerators
			self.sensors['CV'] = open('/sys/bus/i2c/drivers/ina3221x/1-0041/iio:device1/in_current0_input', 'r')
			# Memory
			self.sensors['VDDRQ'] = open('/sys/bus/i2c/drivers/ina3221x/1-0041/iio:device1/in_current1_input', 'r')
			# Other components of the board (eMMC, video, audio, etc)
			self.sensors['SYS5V'] = open('/sys/bus/i2c/drivers/ina3221x/1-0041/iio:device1/in_current2_input', 'r')

		# Launch threads.
		self.measuring = False
		self.executing = True
		threading.Thread.__init__(self)

		# self.reader_thread = threading.Thread(target=self.measure_function)
		# self.reader_thread.start()
		#self.writer_thread = threading.Thread(target=self.write_function)

		# Open output file.
		#self.file = open(os.path.join(self.out_dir, self.filename), 'w')

	#def measure_function(self):
	def run(self):
		while self.executing:
			t = time.time()
			while self.measuring:
				# data = str(time.time())
				# # if self.mode == 'all':
				# # 	for key in self.sensors.keys():
				# # 		data = data + " ; " + self.sensors[key].read().strip()
				# # 		self.sensors[key].seek(0)
				# # else:
				# data = data + " ; " + self.sensors[self.mode].read().strip()
				# t = time.time()
				# self.sensors[self.mode].seek(0)
				# self.queue.put(data)
				# while (time.time() - t) < self.sleep_time:
				# 	continue
				if self.steps > self.discarded: # Discarded samples
					#t = time.time()
					energy = int(self.sensors[self.mode].read().strip())
					tim = time.time() - t
					self.total_energy = self.total_energy + (tim * energy)
					self.min_energy = min(self.min_energy, energy)
					self.max_energy = max(self.max_energy, energy)
					self.time = self.time + tim
					self.sensors[self.mode].seek(0)
					t = time.time() #todo original
					time.sleep(self.sleep_time)

			time.sleep(self.sleep_time)


	# def write_function(self):
	# 	while self.measuring or not self.queue.empty():
	# 		if not self.queue.empty():
	# 			self.file.write(self.queue.get() + '\n')

	def start_measuring(self):
		self.measuring = True
		self.steps = self.steps + 1
		#self.reader_thread.start()
		#self.writer_thread.start()

	def stop_measuring(self):
		self.measuring = False

	def finish(self):
		self.measuring = False
		self.executing = False
		self.join()

		self.total_energy = self.total_energy / (self.steps - self.discarded)
		self.time = self.time / (self.steps - self.discarded)
		#self.writer_thread.join()
		#self.sensors['GPU'].close()

if __name__ == '__main__':
    measurer = EnergyMeter('xavier', 100, 'GPU', '/tmp/', 'prueba.txt')
    measurer.start()
    time.sleep(5)
    measurer.finish()
    print(measurer.total_energy)

