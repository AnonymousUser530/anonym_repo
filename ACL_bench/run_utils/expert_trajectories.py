import pickle

def load_human_expert_trajectory(folder_path, human_alp=None):
    # 1 - loading the trajectory
    try:
        gmms_means, gmms_covs, gmms_mean_rew = pickle.load(open(os.path.join(folder_path, 'human_made_curriculum.pkl'), "rb"))
    except:
        print('Unable to load expert trajectory data: {}'.format(folder_path))
        exit(1)
    gmms_mean_rew = np.interp(gmms_mean_rew, (-150, 350), (0, 1))
    if human_alp is not None: #change alp
        print(gmms_means)
        for gmm_means in gmms_means:
            for gaussian_means in gmm_means:
                gaussian_means[-1] = human_alp/100.0
        print('after')
        print(gmms_means)
    return gmms_means, gmms_covs, gmms_mean_rew

def load_expert_trajectory(folder_path, alp_thr=0.1, max_steps=10000000):
    # 1 - loading the trajectory
    data = {}
    # select seeded run
    try:
        env_params_dict = pickle.load(open(os.path.join(folder_path, 'env_params_save.pkl'), "rb"))
    except:
        print('Unable to load expert trajectory data: {}'.format(folder_path))
        exit(1)
    for k, v in env_params_dict.items():
        if k == 'means' or k == 'covariances' or k == 'weights' or k == 'env_train_len' or k == 'episodes' \
           or k == 'env_train_rewards':
            data[k] = v

    # 2 - pre-processing expert trajectory
    # removing low-alp gaussians
    processed_gmms_means = []
    processed_gmms_covs = []
    processed_gmms_mean_rew = []
    idx_removed_gmms = []
    step_nb = 0
    gmm_step = data['episodes'][0]
    for i, (gmm_means, gmm_covs, episode) in enumerate(zip(data["means"], data["covariances"], data['episodes'])):
        step_nb += sum(data['env_train_len'][i*gmm_step:(i+1)*gmm_step])
        processed_gmm_means = []
        processed_gmm_covs = []
        all_rewards = np.interp(data['env_train_rewards'][episode:episode + gmm_step], (-150, 350), (0, 1))  # from gmm
        rewards = all_rewards[-50:]  # consider mean reward after some training on the GMM
        mean_reward = np.mean(rewards)
        for j, (means, covs) in enumerate(zip(gmm_means, gmm_covs)):
            if means[-1] > alp_thr:  # last mean is ALP dimension
                # add gaussian
                processed_gmm_means.append(means)
                processed_gmm_covs.append(covs)
        if not processed_gmm_means == []:  # gmm not empty after pre-process, lets add it
            processed_gmms_means.append(processed_gmm_means)
            processed_gmms_covs.append(processed_gmm_covs)
            processed_gmms_mean_rew.append(mean_reward)
        else:
            idx_removed_gmms.append(i)
        if step_nb > max_steps:
            break
    print('idx of removed gmms ({}/{}) in expert traj: {}'.format(len(data['means']) - len(processed_gmms_means),
                                                                  len(data['means']),
                                                                  idx_removed_gmms))
    print('number of steps: {}'.format(step_nb))
    return processed_gmms_means, processed_gmms_covs, processed_gmms_mean_rew