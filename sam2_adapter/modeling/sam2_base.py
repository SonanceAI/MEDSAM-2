from sam2.modeling.sam2_base import SAM2Base as SAM2BaseOriginal


class SAM2Base(SAM2BaseOriginal):
    # overload load_state_dict to handle missing_keys and unexpected_keys
    def load_state_dict(self, state_dict):
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False)

        missing_keys = [key for key in missing_keys if 'adapter' not in key]

        if missing_keys:
            raise RuntimeError('Missing keys: {}'.format(missing_keys))
        if unexpected_keys:
            raise RuntimeError('Unexpected keys: {}'.format(unexpected_keys))
        return missing_keys, unexpected_keys
