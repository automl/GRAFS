def replace_ac_function(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_ac_function(module, old, new)

        if isinstance(module, old):
            setattr(model, n, new)
