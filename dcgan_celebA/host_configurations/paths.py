def get_host_paths(data_set):
    root_dict = {}
    if data_set == 'celebA':
        root_dict['data_set'] = '/home/firas/nets/Paper-Implementations/BEGAN/data/CelebA'
        root_dict['attr'] = '/home/alon-ran/Alon/data_sets/celebA'
    return root_dict