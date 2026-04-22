from ml.pipeline.memoization import MemoizationService

def test_memoization_key_changes_when_seed_changes():
    """ Test sul controllo di uguaglianza delle chiavi per il riutilizzo/non riutilizzo degli steps """
    
    memo = MemoizationService()

    key1, _ = memo.build_key(step_name='train_rf', 
                             data_descriptor={'dataset_hash': 'abc'}, 
                             config_descriptor={'max_features': 1000}, 
                             params_descriptor={'model': 'rf'}, 
                             random_descriptor={'seed': 42})
    
    key2, _ = memo.build_key(step_name='train_rf', 
                             data_descriptor={'dataset_hash': 'abc'}, 
                             config_descriptor={'max_features': 1000}, 
                             params_descriptor={'model': 'rf'}, 
                             random_descriptor={'seed': 7})
    
    assert key1 != key2