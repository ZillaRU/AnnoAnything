import os
import ctypes as ct
import numpy as np
import time
import threading

SGTypeTuple = (
   (np.float32, 0),
   (np.int32, 6),
   (np.uint32, 7),
   (np.int8, 2),
   (np.uint8, 3),
   (np.float16, 1)
)
def sglen(t):
    if t in [2,3]:
        return 1
    elif t in [0,6,7]:
        return 4
    elif t == 1:
        return 2

def sgtype(t):
    for nt, bt in SGTypeTuple: 
        if t == nt:
            return bt
def nptype(t):
    for nt, bt in SGTypeTuple: 
        if t == bt:
            return nt

class SGTensor(ct.Structure):
    _fields_ = [
        ("dims", ct.c_uint32),
        ("shape", ct.c_uint32*8),
        ("dtype", ct.c_uint32),
        ("data", ct.c_void_p),
    ]
    def to_numpy(self):
        print(self.dims, self.shape[0:self.dims], self.dtype)
        shape = self.shape[0:self.dims]
        dtype = nptype(self.dtype)
        mem_size = np.prod(shape)*sglen(self.dtype)
        data_ptr = (ct.c_byte*mem_size)()
        ct.memmove(data_ptr, self.data, mem_size)
        #data_ptr = ct.cast(self.data, ct.POINTER(ct.c_byte*mem_size)).contents
        buffer = data_ptr
        return np.frombuffer(buffer, dtype = dtype).reshape(shape)
        
    def from_numpy(self, data):
        self.dims = ct.c_uint32(len(data.shape))
        for i in range(self.dims):
            self.shape[i] = ct.c_uint32(data.shape[i])
        self.dtype = ct.c_uint32(sgtype(data.dtype))
        self.data = data.ctypes.data_as(ct.c_void_p)

class BlobInfo(ct.Structure):
    _fields_ = [
        ("name", ct.c_char_p),
        ("dims_num", ct.c_int),
        ("dtype", ct.c_int),
        ("dims", ct.c_int * 8),
        ("scale", ct.c_float)]

class SGInfer:
    __lib = None

    def __init__(self, bmodel_path, batch=1, devices=None):
        self.bmodel_path = bmodel_path
        if self.__class__.__lib is None:
            lib_path = os.path.join(os.path.dirname(__file__), "libpipeline.so")
            self.__class__.__lib = ct.cdll.LoadLibrary(lib_path)
        self.__lib = self.__class__.__lib
        if devices is not None:
            device_ids = (ct.c_int*len(devices))(*devices)
            device_num = ct.c_int(len(devices))
            self.__lib.runner_use_devices(device_ids, device_num)
        self.runner_id = self.__lib.runner_start_with_batch(ct.c_char_p(bytes(bmodel_path, encoding='utf-8')), batch)
        if devices is not None:
            device_num = ct.c_int(0)
            self.__lib.runner_use_devices(device_ids, device_num)

    @classmethod
    def available_devices(cls):
        if cls.__lib is None:
            lib_path = os.path.join(os.path.dirname(__file__), "libpipeline.so")
            cls.__lib = ct.cdll.LoadLibrary(lib_path)
        max_num = ct.c_int(1024);
        devices = (ct.c_int*max_num.value)()
        real_num = cls.__lib.available_devices(devices, max_num)
        return tuple(devices[i] for i in range(real_num))

    def get_input_info(self):
        num = ct.c_uint32(0)
        self.__lib.get_input_info.restype = ct.POINTER(BlobInfo)
        infos = self.__lib.get_input_info(self.runner_id, ct.byref(num))
        result = dict()
        for _, info in zip(range(num.value), infos):
            result[info.name.decode()] = dict(
                scale=info.scale,
                dtype=info.dtype,
                shape=[info.dims[i] for i in range(info.dims_num)])
        self.__lib.release_input_info(self.runner_id, infos)
        return result

    def get_output_info(self):
        num = ct.c_uint32(0)
        self.__lib.get_output_info.restype = ct.POINTER(BlobInfo)
        infos = self.__lib.get_output_info(self.runner_id, ct.byref(num))
        result = dict()
        for _, info in zip(range(num.value), infos):
            result[info.name.decode()] = dict(
                scale=info.scale,
                dtype=info.dtype,
                shape=[info.dims[i] for i in range(info.dims_num)])
        self.__lib.release_input_info(self.runner_id, infos)
        return result

    def __del__(self):
        self.__lib.runner_stop(self.runner_id)

    def put(self, *inputs):
        if not inputs:
            self.__lib.runner_join(self.runner_id)
            return
        input_num = ct.c_int(len(inputs))
        sg_inputs = (SGTensor*len(inputs))()
        inputs = [i if i.data.c_contiguous else np.ascontiguousarray(i) for i in inputs]
        for i in range(len(inputs)):
            sg_inputs[i].from_numpy(inputs[i])
        task_id = self.__lib.runner_put_input(self.runner_id, input_num, sg_inputs, 1)
        return task_id
        
    def get(self):
        return self.__get(self.__lib.runner_get_output)

    def try_get(self):
        return self.__get(self.__lib.runner_try_to_get_output)

    def stopped(self):
        return self.__lib.runner_all_stopped(self.runner_id)
        
    def empty(self):
        return self.__lib.runner_empty(self.runner_id)

    def infer_all(self, samples, key_func=None, out_func=None, in_func=None):
        self.sample_count = len(samples)
        if self.sample_count == 0:
            return
        self.map_lock = threading.Lock()
        self.finish_cond = threading.Condition()
        self.task_map = {}
        self.sample_index = {}
        self.out_func = out_func
        self.wait_thread = threading.Thread(target=self.__wait_result)
        self.wait_thread.start()
        for i, sample in enumerate(samples):
            sample_id = key_func(i, sample) if key_func else i
            if in_func is not None:
                sample = in_func(sample)
            task_id = self.put(*sample)
            self.map_lock.acquire()
            self.sample_index[sample_id] = len(self.task_map)
            self.task_map[task_id] = sample_id
            self.map_lock.release()
        self.wait_thread.join()
        return self.final_outputs

    def __wait_result(self):
        cached_outputs = []
        while self.sample_count>0:
            task_id, outputs, valid = self.try_get()
            if task_id == 0:
                time.sleep(0.0001)
                continue

            self.map_lock.acquire()
            sample_id = self.task_map[task_id]
            del self.task_map[task_id]
            self.map_lock.release()

            if self.out_func is not None:
                outputs = self.out_func(sample_id, outputs)
            cached_outputs.append((sample_id, outputs))
            self.sample_count -= 1

        self.final_outputs = [None]*len(cached_outputs)
        for id, out in cached_outputs:
            self.final_outputs[self.sample_index[id]] = out


    def __get(self, func):
        output_num= ct.c_uint32(0)
        task_id = ct.c_uint32(0)
        output_valid = ct.c_uint32(0)
        func.restype = ct.POINTER(SGTensor)
        output_tensors = func(self.runner_id, ct.byref(task_id), ct.byref(output_num), ct.byref(output_valid))
        if(task_id.value == 0):
            return 0, [], 0
        outputs = []
        if output_valid.value == 0:
            return task_id.value, [], False
        for i in range(output_num.value):
            outputs.append(output_tensors[i].to_numpy())
        self.__lib.runner_release_output(output_num, output_tensors)
        return task_id.value, outputs, True

    def infer_one(self, *inputs):
        in_task_id = self.put(*inputs)
        out_task_id, outputs, valid = self.get()
        assert in_task_id == out_task_id
        return outputs, valid

    def wait_to_stop(self):
        while not self.stopped():
            self.put()
            time.sleep(0.01)

    def get_durations(self):
        num = ct.c_uint32(0)
        self.__lib.get_runner_durations.restype = ct.POINTER(ct.c_uint32)
        durations = self.__lib.get_runner_durations(self.runner_id, ct.byref(num))
        result = [durations[i] for i in range(num.value)]
        self.__lib.release_unsigned_pointer(durations);
        return result

    def show(self):
        self.__lib.runner_show_status(self.runner_id)


if __name__ == "__main__":

    n = np.arange(1*3*2*2).astype(np.float32).reshape(1,3,2,2)
    t = SGTensor()
    t.from_numpy(n)
    nn = t.to_numpy()
    n[0,0,0,0]=2.4
    print(n)
    print(nn)
    bmodel_path = os.path.join(os.path.dirname(__file__), "test_model/compilation.bmodel")
    print("devices=",SGInfer.available_devices())
    s = SGInfer(bmodel_path)
    i = np.arange(1*3*20*20).astype(np.float32).reshape(1,3,20,20)
    print(i)
    print(s.infer_one(i))
    print(s.infer_all([i]*3))
    print(s.infer_all([i]*4))
    s2 = SGInfer(bmodel_path, 1, (0,))
    print(s2.infer_one(i))
