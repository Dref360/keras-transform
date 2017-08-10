def get_value(tree, idx):
    if not idx or not isinstance(tree, (list, tuple)):
        return tree
    elif len(idx) == 1:
        return tree[idx[0]]
    else:
        return get_value(tree[idx[0]], idx[1:])


def handle_mask(mask, tree):
    """Expand the mask to match the tree structure.
    :param mask: boolean mask
    :param tree: tree structure
    :return: boolean mask
    """
    if isinstance(mask, bool):
        return [mask] * len(tree)
    return mask


def apply_fun(tree, fun, mask, **kwargs):
    """Apply a function recursively on a list.
    :param tree: Tree structure of lists
    :param fun: function to apply
    :param mask: boolean mask to control the application of `fun`.
    :param kwargs: arguments for `fun`
    :return: list
    """
    if not isinstance(tree, (list, tuple)):
        return fun(tree, **kwargs) if mask else tree
    else:
        return [apply_fun(tr, fun, ma, **kwargs) if ma else tr for tr, ma in zip(tree, handle_mask(mask, tree))]
