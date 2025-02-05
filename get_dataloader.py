
from mnist import getMnistDataset
from torch.utils.data import DataLoader, Dataset

def build_dataloader(phase, dataset, batch_size, train_shuffle=True):
    if phase == "train":
        data_dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            # shuffle=True if not customSamplerFlag else False,
            shuffle=train_shuffle,
            num_workers=1,
            # num_workers=2,
            # sampler=RandomSampler(dataset, num_samples=5000) if customSamplerFlag else None
        )
    elif phase == "val":
        data_dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1
            # num_workers=2
        )
    elif phase == "test":
        data_dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1
            # num_workers=2
        )
    return data_dataloader
    

def getDataset(
        dataset_name, phase, img_size, norm_type="n2"
    ):
    
    if dataset_name == "mnist":
        d = getMnistDataset(phase, img_size, norm_type)
    else:
        raise RuntimeError(f"dataset_name:{dataset_name} is invalid")
    return d


def getDataloader(
        dataset_name, phase, img_size, batch_size, norm_type="n2"
    ):
    """
    get dataloader according to the params
    Args:
        dataset_name (_type_): _description_
        phase (_type_): train, val, test
        img_size (_type_): size of image
        batch_size (_type_): batch size
    Returns:
        Dataloader
    """
    #region check proportion
    if "mnist" in dataset_name:
        assert phase in ["train", "val", "test"]
    else:
        raise RuntimeError(f"dataset_name:{dataset_name} is invalid!")
    #endregion
    
    d = getDataset(
        dataset_name, phase, img_size, norm_type
    )

    return build_dataloader(
        phase=phase, dataset=d, 
        batch_size=batch_size
    )

