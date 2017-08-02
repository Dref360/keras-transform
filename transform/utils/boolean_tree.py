def get_value(tree, idx):
    if not idx or not isinstance(tree, (list, tuple)):
        return tree
    elif len(idx) == 1:
        return tree[idx[0]]
    else:
        return get_value(tree[idx[0]], idx[1:])


def handle_mask(mask, tree):
    if isinstance(mask, bool):
        return [mask] * len(tree)
    return mask


def apply_fun(tree, fun, mask,**kwargs):
    if not isinstance(tree, (list, tuple)):
        return fun(tree,**kwargs) if mask else tree
    else:
        return [apply_fun(tr,fun,ma,**kwargs) if ma else tr for tr, ma in zip(tree, handle_mask(mask, tree))]
