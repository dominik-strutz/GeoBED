import pytest

import torch

from geobed.continuous.core import *

TEST_M_PRIOR_DIST = torch.distributions.MultivariateNormal(torch.zeros(3), torch.eye(3))
TEST_MODEL_SAMPLES = torch.linspace(0, 1, 30).reshape(10,3)

TEST_NUISANCE_PRIOR_DIST = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
TEST_NUISANCE_SAMPLES = torch.linspace(0, 1, 100).reshape(50, 2)

TEST_DESIGN = torch.ones(2).unsqueeze(-1)

def dummy_forward_function(design, model_samples):
    
    out = model_samples.sum(-1).unsqueeze(0)
    return design * out

def dummy_forward_function_with_dict(design, model_samples):
    return {'data': model_samples, 'other': torch.ones(model_samples.shape)}

def dummy_forward_function_with_nuisance(design, model_samples, nuisance_samples=None):

    out = (model_samples.sum(-1) + nuisance_samples.sum((-2, -1))).unsqueeze(0)   
    return design * out
    

def test_entropy():

    # test for distribution with analytical entropy
    analytical_entropy = torch.distributions.Normal(0,1).entropy()
    ananlytical_test_class = BED_Class(
        forward_function = dummy_forward_function,
        m_prior_dist=torch.distributions.Normal(0,1),
    )

    assert ananlytical_test_class.m_prior_dist_entropy == analytical_entropy

    # test for distribution with no analytical entropy
    dist = torch.distributions.MixtureSameFamily(
        mixture_distribution=torch.distributions.Categorical(torch.tensor([0.5, 0.5])),
        component_distribution=torch.distributions.Normal(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])),
    )

    estimated_test_class = BED_Class(
        forward_function = dummy_forward_function,
        m_prior_dist=dist,
    )
    assert pytest.approx(estimated_test_class.m_prior_dist_entropy, rel=0.01) == analytical_entropy

    # test for distribution with no log_prob
    sample_dist = SampleDistribution(TEST_MODEL_SAMPLES)

    sample_dist_test_class = BED_Class(
        forward_function = dummy_forward_function,
        m_prior_dist=sample_dist,
    )
    assert sample_dist_test_class.m_prior_dist_entropy == 0

def test_get_m_prior_sample():
    
    # test defined distribution
    test_class = BED_Class(
        forward_function = dummy_forward_function,
        m_prior_dist=TEST_M_PRIOR_DIST
    )
    
    out_1 = test_class.get_m_prior_samples(torch.Size([10,4]), random_seed=0)
    
    torch.manual_seed(0)
    test_samples = TEST_M_PRIOR_DIST.sample((10,4))
    
    assert out_1.tolist() == test_samples.tolist()
    
    # test predefined samples
        
    sd_test_class = BED_Class(
        forward_function = dummy_forward_function,
        m_prior_dist=TEST_MODEL_SAMPLES,
    )
    
    out_2 = sd_test_class.get_m_prior_samples(5)
        
    assert out_2.shape == (5,3)
    assert out_2.tolist() == TEST_MODEL_SAMPLES[:5].tolist()
    
    out_3 = sd_test_class.get_m_prior_samples((2,2))
    
    assert out_3.shape == (2,2,3)
    assert out_3.tolist() == TEST_MODEL_SAMPLES[5:9].reshape(2,2,3).tolist()
    
    with pytest.raises(ValueError):
        sd_test_class.get_m_prior_samples(10)

def test_get_nuisance_prior_samples():
        
    # test undefined nuisance distribution
    test_class_0 = BED_Class(
        forward_function = dummy_forward_function,
        m_prior_dist=TEST_M_PRIOR_DIST,
    )
    
    assert test_class_0.get_nuisance_prior_samples(TEST_MODEL_SAMPLES, 2) == None
    
    # test defined unconditional distribution
    test_class_1 = BED_Class(
        forward_function = dummy_forward_function,
        m_prior_dist=TEST_M_PRIOR_DIST,
        nuisance_dist=TEST_NUISANCE_PRIOR_DIST,
    )
    
    out_1 = test_class_1.get_nuisance_prior_samples(TEST_MODEL_SAMPLES, 2, random_seed=0)
    torch.manual_seed(0)
    test_samples_1 = TEST_NUISANCE_PRIOR_DIST.sample((2,10)).swapaxes(0,1)
        
    assert out_1.tolist() == test_samples_1.tolist()
    
    # test defined conditional distribution

    def conditional_nuisance_dist(x):
        return torch.distributions.MultivariateNormal(x, torch.eye(x.shape[-1]))

    test_class_2 = BED_Class(
        forward_function = dummy_forward_function,
        m_prior_dist=TEST_M_PRIOR_DIST,
        nuisance_dist=conditional_nuisance_dist,
    )
    out_2 = test_class_2.get_nuisance_prior_samples(TEST_MODEL_SAMPLES, 2, random_seed=0)
    torch.manual_seed(0)
    test_samples_2 = conditional_nuisance_dist(TEST_MODEL_SAMPLES).sample((2,)).swapaxes(0,1)  
    assert out_2.tolist() == test_samples_2.tolist()
    
    # test defined unconditional predefined samples
    
    test_class_3 = BED_Class(
        forward_function = dummy_forward_function,
        m_prior_dist=TEST_M_PRIOR_DIST,
        nuisance_dist=TEST_NUISANCE_SAMPLES,
    )
    out_3 = test_class_3.get_nuisance_prior_samples(TEST_MODEL_SAMPLES, 2)    
    assert out_3.tolist() == TEST_NUISANCE_SAMPLES[:20].reshape(2, 10, 2).swapaxes(0,1).tolist()
    
    with pytest.raises(ValueError):
        test_class_3.get_nuisance_prior_samples(TEST_MODEL_SAMPLES, 100)

def test_get_forward_function_samples():

    # test forward function with no nuisance parameters
    test_class_1 = BED_Class(
        forward_function = dummy_forward_function,
        m_prior_dist=TEST_MODEL_SAMPLES,
    )
    out_1_1 = test_class_1.get_forward_function_samples(
        design=TEST_DESIGN,
        n_samples=2)

    assert out_1_1.shape == (2,2)    
    assert out_1_1.tolist() == (TEST_DESIGN * TEST_MODEL_SAMPLES[:2].sum(-1).unsqueeze(0)).tolist()

    # test forward function wit no nuisance parameters and return dict
    out_1_2_data, out_1_2_model, out_1_2_nuisance = test_class_1.get_forward_function_samples(
        design=TEST_DESIGN,
        n_samples=2,
        return_parameter_samples=True
    )
    
    assert out_1_2_data.tolist() == (TEST_DESIGN * TEST_MODEL_SAMPLES[2:4].sum(-1).unsqueeze(0)).tolist()
    assert out_1_2_nuisance == None
    assert out_1_2_model.tolist() == TEST_MODEL_SAMPLES[2:4].tolist()
    
    # test forward function with nuisance parameters
    test_class_2 = BED_Class(
        forward_function = dummy_forward_function_with_nuisance,
        m_prior_dist=TEST_MODEL_SAMPLES,
        nuisance_dist=TEST_NUISANCE_SAMPLES,
    )
    out_2_1 = test_class_2.get_forward_function_samples(
        design=TEST_DESIGN,
        n_samples_model=2,
        n_samples_nuisance=2
    )
    
    assert out_2_1.tolist() == dummy_forward_function_with_nuisance(
        design=TEST_DESIGN,
        model_samples=TEST_MODEL_SAMPLES[:2],
        nuisance_samples=TEST_NUISANCE_SAMPLES[0:4].reshape(2,2,2).swapaxes(0,1)
    ).tolist()

    out_2_2_data, out_2_2_model, out_2_2_nuisance = test_class_2.get_forward_function_samples(
        design=TEST_DESIGN,
        n_samples_model=2,
        n_samples_nuisance=2,
        return_parameter_samples=True
    )
    
    assert out_2_2_data.tolist() == dummy_forward_function_with_nuisance(
        design=TEST_DESIGN,
        model_samples=TEST_MODEL_SAMPLES[2:4],
        nuisance_samples=TEST_NUISANCE_SAMPLES[4:8].reshape(2,2,2).swapaxes(0,1)
    ).tolist()
    assert out_2_2_model.tolist() == TEST_MODEL_SAMPLES[2:4].tolist()
    assert out_2_2_nuisance.tolist() == TEST_NUISANCE_SAMPLES[4:8].reshape(2,2,2).swapaxes(0,1).tolist()



if __name__ == "__main__":
    test_entropy()
    test_get_m_prior_sample()
    test_get_nuisance_prior_samples()
    test_get_forward_function_samples()
    