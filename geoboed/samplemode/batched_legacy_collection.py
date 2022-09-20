        # if preload_samples:
        #     # tolist necessary as as single elemet tensor will return an int instead of a tensor
        #     samples = np.concatenate([pre_samples[:n_final_samples, design_i.tolist()][:, None, :] for design_i in design_list], axis=1)            
        # else:
        #     samples = np.concatenate([dataframe['data'][:n_final_samples, design_i.tolist()][:, None, :] for design_i in design_list], axis=1)            
        # samples = torch.tensor(samples)
        
        # #TODO look for way to make noise consistent when batched
        # if var_guide == 'gmm':
        #     guide = GaussianMixtureModel_batched(samples, **guide_args)
        # elif var_guide == 'nf_cpl': 
        #     guide = Coupling_Spline_NF_batched(samples, **guide_args) 
        # elif type(var_guide) == str:
        #     raise NotImplementedError('Guide not implemented for batched calculations.')
        
        # optimizer = optimizer_constructor(guide)
        # if scheduler is not None:
        #     scheduler = scheduler_constructor(optimizer)
        
        # losses = []
        
        # for step in (tqdm(
        #     range(n_steps), total=n_steps, disable=disable_tqdm, leave=True)):
            
        #     optimizer.zero_grad()
            
        #     if stoch_sampling:
        #         random_indices = random.sample(range(n_samples), n_stochastic)
        #         x = self.data_likelihood(samples[random_indices]).sample([1, ]).flatten(start_dim=0, end_dim=1)
                
        #     else:
        #         x = self.data_likelihood(samples[:n_samples]).sample([1, ]).flatten(start_dim=0, end_dim=1)                             
            
        #     loss = -guide.log_prob(x).mean(dim=0)
            
        #     loss.backward(torch.ones_like(loss))
        #     optimizer.step()
        #     guide.clear_cache()
        
        #     if return_dict:
        #         losses.append(loss.detach().tolist())

        # #TODO: implement option to not reuse samples for final samples  

        # likeliehood = self.data_likelihood(samples)
        # samples = likeliehood.sample([1, ]).flatten(start_dim=0, end_dim=1)

        # marginal_lp = guide.log_prob(samples)     
        # conditional_lp = likeliehood.log_prob(samples)
                
        # eig = (conditional_lp - marginal_lp).sum(0) / n_final_samples 
        
        # eig_list = eig.detach().tolist()
                        
                        
        # if return_dict:
        #     output_dict = {
        #         'var_guide_name':  var_guide,
        #         'var_guide'     :  guide,
        #         'n_steps':         n_steps,
        #         'n_samples':       n_samples,
        #         'n_final_samples': n_final_samples,
        #         'stoch_sampling':  stoch_sampling,
        #         'n_stochastic':    n_stochastic,
        #         'optim':           optim,
        #         'scheduler':       scheduler,
        #         'batched':         batched,
        #         'guide_args':      guide_args,
        #         'losses':          np.array(losses),
        #         }
        #     return np.array(eig_list), output_dict
        # else:
        #     return np.array(eig_list), None
        
        
        
        
        
        
        # #TODO: Cant train the same nn for all designs, obiously ...
        
        # if design_selection:
        #     samples = torch.tensor(dataframe['data'][ :n_samples, design_selection])
        #     model_space = torch.tensor(dataframe['prior'][:n_samples])

        # else:
        #     samples = torch.tensor(dataframe['data'][ :n_samples, :])
        #     model_space = torch.tensor(dataframe['prior'][:n_samples])

        # # # set random set so each design point has same noise realisation
        # # pyro.set_rng_seed(0)
                
        # if var_guide == 'mdn':
        #     guide = MDN_guide_batched(samples, model_space, **guide_args)
        # elif type(var_guide) == str:
        #     raise NotImplementedError('Guide not implemented')
        
        # optimizer = optimizer_constructor(guide)
        # if scheduler is not None:
        #     scheduler = scheduler_constructor(optimizer)        
        # losses = []

        # for step in (pbar := tqdm(range(n_steps), total=n_steps, disable=False, leave=False)):
            
        #     optimizer.zero_grad()
            
        #     if stoch_sampling:
        #         random_indices = random.sample(range(n_samples), n_stochastic)
        #         x = self.data_likelihood(samples[random_indices]).sample([1, ]).flatten(start_dim=0, end_dim=1)

        #         ln_p_x2_given_x1 = guide.log_prob(model_space[random_indices], x)
        #     else:
        #         x = self.data_likelihood(samples).sample([1, ]).flatten(start_dim=0, end_dim=1)    
        #         ln_p_x2_given_x1 = guide.log_prob(model_space, x)
            
        #     loss = -(ln_p_x2_given_x1.mean(dim=0))

        #     loss.backward(torch.ones_like(loss)) 
        #     optimizer.step()
        #     guide.clear_cache()

        #     if scheduler is not None:
        #         if step % scheduler_n_iterations == 0:
        #             scheduler.step()

        #     losses.append(loss.detach().tolist())

        #     pbar.set_description(f"Loss: {loss.sum().detach().item():.2e}, lr: {optimizer.param_groups[0]['lr']:.2e}")
            
        # if design_selection:
        #     final_samples = torch.tensor(dataframe['data'][ :n_final_samples, design_selection])
        #     likeliehood = self.data_likelihood(final_samples, dataframe['design'][design_selection])
        # else:
        #     final_samples = torch.tensor(dataframe['data'][ :n_final_samples, :])
        #     likeliehood = self.data_likelihood(final_samples, dataframe['design'][:])
        
        # model_space = torch.tensor(dataframe['prior'][:n_final_samples])

        # final_samples = likeliehood.sample([1, ]).flatten(start_dim=0, end_dim=1)
        
        # #TODO: should be the same as loss without minus sign use directly for faster evaluations
        # marginal_lp = guide.log_prob(model_space, final_samples)
        
        # eig = prior_ent + -(-marginal_lp).sum(0) / n_samples 
        
        # eig_list = eig.detach().tolist()                  
        
        # if return_guide:
        #     var_list = guide
        
        
        # class MDN_guide_batched(VarPost_Base):
    
    # def __init__(self, data_samples, model_sample, K=3):
    #     super(MDN_guide_batched, self).__init__()
        
    #     self.K = K
    #     self.design_dim = data_samples.shape[-2]

    #     self.data_dim = data_samples.shape[-1]
    #     self.model_dim = model_sample.shape[-1]

    #     self.nn_N = 128

    #     self.ds_means = data_samples.mean(dim=0)
    #     self.ds_stds  = data_samples.std(dim=0)
                
    #     self.ms_means = model_sample.mean(dim=0, keepdim=True)
    #     self.ms_stds  = model_sample.std(dim=0, keepdim=True)
        
    #     self.z_1 = nn.Linear(self.data_dim, self.nn_N)
    #     self.z_2 = nn.Linear(self.nn_N, self.nn_N)
    #     # self.z_3 = nn.Linear(self.nn_N, self.nn_N)

    #     self.h_alpha = nn.Linear(self.nn_N, self.K)
    #     self.h_mu =    nn.Linear(self.nn_N, self.K*self.model_dim)
    #     self.h_sigma = nn.Linear(self.nn_N, self.K*self.model_dim)
        
    # def evaluate(self, data_samples):
        
    #     batch_dim = data_samples.shape[:-1]
    #     data_samples = data_samples.unsqueeze(1)
        
    #     data_samples -= self.ds_means
    #     data_samples /= self.ds_stds        
        
    #     z_h = torch.tanh(self.z_1(data_samples))
    #     z_h = torch.tanh(self.z_2(z_h))
    #     # z_h = torch.tanh(self.z_3(z_h))
        
    #     alpha = nn.functional.log_softmax(self.h_alpha(z_h), dim=-1)
    #     mu    = self.h_mu(z_h)
    #     sigma = torch.nn.functional.elu(self.h_sigma(z_h)) + 1.0 + 1e-15

    #     alpha = alpha.reshape((*batch_dim, self.K))
    #     mu    = mu.reshape((*batch_dim, self.K, self.model_dim))
    #     sigma = sigma.reshape((*batch_dim, self.K, self.model_dim))
        
    #     mix = dist.Categorical(logits=alpha)
    #     comp = dist.Normal(mu , sigma).to_event(1)
    #     dit = dist.MixtureSameFamily(mix, comp)
        
    #     normalizing = torch_transforms.AffineTransform(
    #         self.ms_means, self.ms_stds, event_dim=1)
        
    #     dit = dist.TransformedDistribution(dit, normalizing)
        
    #     return alpha, mu, sigma, dit
        
    # def log_prob(self, model_samples, data_samples):
        
    #     _, _, _, dit = self.evaluate(data_samples)           
        
    #     return dit.log_prob(model_samples.unsqueeze(1))

    # def clear_cache(self):
    #     pass
    
    # def sample(self, data_samples):
    #     #TODO make sampling a bit more transparent with how many samples should be drawn from each posterior individually

    #     _, _, _, dit = self.evaluate(data_samples)
                
    #     sample = dit.sample([1,]).flatten(start_dim=0, end_dim=1)
    #     return sample
    
    # def get_mix(self, data_samples):
    #     alpha, _, _, _ = self.evaluate(data_samples)
    #     return alpha
    
    # def get_mean(self, data_samples):
    #     _, mu, _, _ = self.evaluate(data_samples)
    #     return mu
    
    # def get_sigma(self, data_samples):
    #     _, _, sigma, _ = self.evaluate(data_samples)
    #     return sigma