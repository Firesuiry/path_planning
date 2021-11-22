def soft_update(target, source, x):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        # temp = copy.deepcopy(target_param.data)
        target_param.data.copy_((1 - x) * target_param.data + x * source_param.data)
        # assert torch.any(target_param.data != temp)