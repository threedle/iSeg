import numpy as np
from tqdm import tqdm


class ObjLoader(object):
    def __init__(self, file_name):
        self.vertices = None
        self.faces = []

        try:
            f = open(file_name)
            num_lines = sum(1 for line in f if line.rstrip())
            f.close()

            self.vertices = np.zeros([num_lines, 6], dtype=np.float32)
            f = open(file_name)
            for line_idx, line in enumerate(tqdm(f, total=num_lines)):
                if line[:2] == "v ":
                    space_idx = [idx + 1 for idx in range(len(line)) if line[idx] == " "]
                    # xyz
                    self.vertices[line_idx, 0:3] = np.array([line[space_idx[0]:space_idx[1]], line[space_idx[1]:space_idx[2]], line[space_idx[2]:space_idx[3]]], dtype=np.float64)

                    # rgb
                    if len(space_idx) > 6:
                        self.vertices[line_idx, 3:6] = np.array([line[space_idx[3]:space_idx[4]], line[space_idx[4]:space_idx[5]], line[space_idx[5]:space_idx[6]]], dtype=np.float64)
                    elif len(space_idx) == 6:
                        self.vertices[line_idx, 3:6] = np.array([line[space_idx[3]:space_idx[4]], line[space_idx[4]:space_idx[5]], line[space_idx[5]:]], dtype=np.float64)

                elif line[0] == "f":
                    string = line.replace("//", "/")

                    i = string.find(" ") + 1
                    face = []
                    for item in range(string.count(" ")):
                        if string.find(" ", i) == -1:
                            face.append(string[i:-1])
                            break
                        face.append(string[i:string.find(" ", i)])
                        i = string.find(" ", i) + 1

                    self.faces.append(tuple(face))

            f.close()
        except IOError:
            print(".obj file not found.")


if __name__ == '__main__':
    # usage example
    obj_file_path = "/path/to/obj/file.obj"
    obj_data = ObjLoader(obj_file_path)

    print(obj_data.vertices.shape)
