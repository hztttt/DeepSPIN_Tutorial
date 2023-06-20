import argparse
import numpy as np
import os
import copy

def read_frames(dir_frames):
    frames_list = []
    frames_count = 0
    for root, dirs, files in os.walk(dir_frames):
        for d in dirs:
            frames_list.append(os.path.abspath(os.path.join(root, d)))
            frames_count += 1
    frames_list = sorted(frames_list, key=lambda x: x.split('/')[-1])
    print('read frames: %d' %frames_count)
    return frames_list

def check_converge(frame, file_log):
    flag_converge = False
    idx_converge = 0
    os.system('tail -200 %s > temp_log' %(frame + '/' + file_log))
    with open('temp_log', 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if 'F' in lines[i] and 'E0' in lines[i] and 'mag' in lines[i]:
                idx_converge = i
                break
        if idx_converge != 0:
            flag_converge = True
    if flag_converge:
        print('%s converged' %frame)
    else:
        print('%s not converge!' %frame)
    return flag_converge, idx_converge

def get_ener_spin(file_name, natoms, idx_converge):
    ener = 0.0
    spin = np.zeros((natoms, 3), dtype=float)
    force = np.zeros((natoms, 3), dtype=float)
    magforce = np.zeros((natoms, 3), dtype=float)
    with open(file_name, 'r') as f:
        lines = f.readlines()
        ener = float(lines[idx_converge + 4].split()[0])
        for i in range(natoms):
            for j in range(3):
                spin[i][j] = float(lines[idx_converge + 8].split()[3 * i + j])
                force[i][j] = float(lines[idx_converge + 12].split()[3 * i + j])
                magforce[i][j] = float(lines[idx_converge + 14].split()[3 * i + j])
    return ener, spin, force, magforce

def get_box_coord(file_name, natoms):
    box = np.zeros((3, 3), dtype=float)
    coord = np.zeros((natoms, 3), dtype=float)
    coord_style = 'Direct'
    with open(file_name, 'r') as f:
        lines = f.readlines()
        box_coeff = float(lines[1].split()[0])
        for i in range(3):
            for j in range(3):
                box[i][j] = float(lines[2+i].split()[j]) * box_coeff
        if 'Cartesian' in lines[7]:
            coord_style = 'Cartesian'
        for i in range(natoms):
            direct = list(map(float, lines[8+i].split()[0:3]))
            direct = np.array(direct).reshape((1, 3))
            if coord_style == 'Direct':
                cartesian = np.matmul(direct, box)
            else:
                cartesian = direct
            coord[i][:] = cartesian[:]
    return box, coord

def get_type_list(file_name, natoms):
    type_list = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
        ntypes = len(lines[6].split())
        for i in range(ntypes):
            type_list.append(int(lines[6].split()[i]))
        assert(sum(type_list) == natoms)
    return type_list

def extend_coord(coord, spin, type_list, ntypes_spin, len_spin):
    endidx_spin = sum(type_list[0 : ntypes_spin])
    spin = spin[:endidx_spin, :]
    spin_norm = np.tile(np.linalg.norm(spin, axis=1, keepdims=True), (1, 3))
    virt_coord = spin / spin_norm * len_spin
    virt_coord += coord[:endidx_spin, :]
    ext_coord = np.concatenate((coord, virt_coord), axis=0)
    return ext_coord

def extend_force(force, magforce, type_list, ntypes_spin):
    endidx_spin = sum(type_list[0 : ntypes_spin])
    magforce = magforce[:endidx_spin, :]
    ext_force = np.concatenate((force, magforce), axis=0)
    return ext_force

def get_raw_type(type_list, ntypes_spin):
    raw_type = []
    ext_type_list = copy.deepcopy(type_list)
    ext_type_list.extend(type_list[:ntypes_spin])
    for i in range(len(ext_type_list)):
        temp_type = np.ones((1, ext_type_list[i]), dtype=int) * i
        raw_type.append(copy.deepcopy(temp_type))
    raw_type = np.concatenate(raw_type, axis=1)
    return raw_type

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_frames', type=str, help='directory of frames to extract data')
    parser.add_argument('--natoms', type=int, help='num of real atoms')
    parser.add_argument('--ntypes', type=int, help='num of types')
    parser.add_argument('--ntypes_spin', type=int, help='num of types with spin')
    parser.add_argument('--len_spin', type=float, nargs="+", help='length of spin from types with spin')
    parser.add_argument('--file_log', type=str, help='log file of DeltaSpin result')
    args = parser.parse_args()

    frames_list = read_frames(args.dir_frames)
    all_box = []
    all_coord = []
    all_ener = []
    all_force = []
    type_list = get_type_list(frames_list[0]+'/POSCAR', args.natoms)
    
    for i in range(len(frames_list)):
        flag_converge, idx_converge = check_converge(frames_list[i], args.file_log)
        if flag_converge:
            temp_box, temp_coord = get_box_coord(frames_list[i]+'/POSCAR', args.natoms)
            temp_ener, temp_spin, temp_force, temp_magforce = get_ener_spin('temp_log', args.natoms, idx_converge)
            ext_coord = extend_coord(temp_coord, temp_spin, type_list, args.ntypes_spin, args.len_spin)
            ext_force = extend_force(temp_force, temp_magforce, type_list, args.ntypes_spin)
            all_box.append(np.reshape(temp_box, (1, -1)))
            all_coord.append(np.reshape(ext_coord, (1, -1)))
            all_ener.append(np.reshape(temp_ener, (1, -1)))
            all_force.append(np.reshape(ext_force, (1, -1)))
            print('frame %d succeed' %(i+1))
        else:
            print('frame %d fail' %(i+1))
        os.system('rm temp_log')
    
    raw_box = np.concatenate(all_box, axis=0)
    raw_coord = np.concatenate(all_coord, axis=0)
    raw_ener = np.concatenate(all_ener, axis=0)
    raw_force = np.concatenate(all_force, axis=0)
    raw_type = get_raw_type(type_list, args.ntypes_spin)
    np.savetxt("box.raw", raw_box, fmt='%10.8f')
    np.savetxt("coord.raw", raw_coord, fmt='%10.8f')
    np.savetxt("energy.raw", raw_ener, fmt='%10.8f')
    np.savetxt("force.raw", raw_force, fmt='%10.8f')
    np.savetxt("type.raw", raw_type, fmt='%d')
    print('get raw files done')


if __name__ == '__main__':
    main()

