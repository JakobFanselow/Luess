import h5py
import numpy as np
from pathlib import Path

class ChessPositionWriter:
    def __init__(self, file_path,buffer_size=65536, removeOld=True):
        self.file_path = file_path
        self.buffer_size = buffer_size
        self.bitposition_buffer = []
        self.label_buffer = []
        if removeOld:
            path = Path(file_path)
            if path.exists():
                path.unlink()

    def add_position(self, position_bitmap, label):
        self.bitposition_buffer.append(position_bitmap)
        self.label_buffer.append(label)

        if len(self.bitposition_buffer) >= self.buffer_size:
            self.flush()


    def flush(self):
        if not self.bitposition_buffer:
            return
        with h5py.File(self.file_path,'a') as f:
            if 'positions' not in f:
                positions_array = np.array(self.bitposition_buffer)  # shape (N, 15)
                f.create_dataset(
                    'positions',
                    data=positions_array,
                    maxshape=(None, positions_array.shape[1]),
                    chunks=(self.buffer_size, positions_array.shape[1]),
                    compression=None
                )

                labels_array = np.array(self.label_buffer).reshape(-1,1)
                f.create_dataset(
                    'labels',
                    data=labels_array,
                    maxshape=(None,1),
                    chunks=(self.buffer_size,1),
                    compression=None
                )

            else:
                positions = f['positions']
                labels = f['labels']
                
                curr_size = positions.shape[0]
                new_size = curr_size + len(self.bitposition_buffer)
                
                positions.resize((new_size,positions.shape[1]))
                labels.resize((new_size, 1))
                
                positions[curr_size:] = self.bitposition_buffer
                labels[curr_size:] = np.array(self.label_buffer).reshape(-1, 1)

        self.bitposition_buffer = []
        self.label_buffer = []