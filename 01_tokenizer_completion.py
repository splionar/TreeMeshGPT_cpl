from fns import process_mesh, sample_point_cloud, quantize_remove_duplicates, prepare_halfedge_mesh, create_io_sequence

"""
Have to adjust the following:
MESH_FN: path to incomplete mesh
AUGMENTATION must be False

Make sure io_dict = tokenize(MESH_FN, MESH_FN_FULL,QUANT_BIT, N_TRIAL, MAX_N_FACES, AUGMENTATION)
is successful (does not skip sample).
"""

QUANT_BIT = 7
MESH_FN_FULL = "demo/objaverse_nut.obj" # not needed
MESH_FN = "demo/objaverse_nut_half.obj"
N_TRIAL = 20
MAX_N_FACES = 5500
AUGMENTATION = False # Always set False for this experiment
    
def manifold_trial(mesh_fn, mesh_fn_full, quant_bit, is_augment):

    # partial mesh
    vertices, triangles = process_mesh(mesh_fn, augment = is_augment) # Open, augment, normalize mesh

    # sample pc from full mesh (not needed for inference)
    vertices_full, triangles_full = process_mesh(mesh_fn_full, augment = is_augment)
    pc = sample_point_cloud(vertices_full, triangles_full, sampling='uniform') 

    vertices, triangles = quantize_remove_duplicates(vertices, triangles, quant_bit = quant_bit) # Quantize and remove duplicates
    o3d_half_edge_mesh, vertices, n_faces = prepare_halfedge_mesh(vertices, triangles)
    return o3d_half_edge_mesh, vertices, n_faces, pc

def tokenize(mesh_fn, mesh_fn_full, quant_bit, n_trial, max_n_faces, is_augment):
    success = False 
    for trial in range(n_trial):  
        # 7-bit quantization with high face count tends to violate manifold condition. Keep trying by using different augmentation if fails.
        try:
            o3d_half_edge_mesh, vertices, n_faces, pc = manifold_trial(mesh_fn, mesh_fn_full, quant_bit, is_augment)
            success = True  
            break  
        except Exception as e:
            continue
        
    if success and n_faces <= max_n_faces:
        edges, gt = create_io_sequence(o3d_half_edge_mesh)
                    
        output_seq_no_aux_token = [elem for sublist in gt for elem in sublist]
        input_seq_no_aux_token = [elem for sublist in edges for elem in sublist]
        
        assert len(output_seq_no_aux_token) == len(input_seq_no_aux_token)
        n_seq = len(output_seq_no_aux_token)
        
        io_dict = {
            'vertices': vertices,
            'edges': edges,
            'n_faces': n_faces,
            'n_seq': n_seq,
            'pc': pc,
            'gt': gt
        }
        
        print("IO creation successful. Mesh face count: {}. Sequence length: {}. Num trials: {}.".format(n_faces, n_seq, trial+1))
        return io_dict
    else:
        print(f"Skipping sample.")
        return None

io_dict = tokenize(MESH_FN, MESH_FN_FULL,QUANT_BIT, N_TRIAL, MAX_N_FACES, AUGMENTATION)

# save io_dict as pkl
import os
import pickle

if io_dict is not None:
    out_path = os.path.join("dummy_completion/incomplete.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(io_dict, f)