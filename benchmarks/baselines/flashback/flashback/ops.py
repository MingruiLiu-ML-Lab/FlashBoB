import numpy as np
from .attentions.flash_sigmoid_op import sigmoid_mha as _sigmoid_mha
from .attentions.flash_softmax_op import softmax_mha as _softmax_mha
from .attentions.naive_softmax_op import naive_softmax_mha as _naive_softmax_mha
from .attentions.naive_sigmoid_op import naive_sigmoid_mha as _naive_sigmoid_mha

from .attentions.flash_sigmoid_op import infer_precision
from .pallas_utils import Precision

def get_default_bias_sm_scale(bias, sm_scale, Q):
    sl = Q.shape[1]
    d = Q.shape[-1]
    if bias == 'default':
        bias = float(np.log(sl) * -5)

    if sm_scale == 'default':
        sm_scale = float(d)**(-0.5)

    return bias, sm_scale

def get_precision(Q, K, V, precision=None):
    if precision is None:
        new_precision = infer_precision(Q, K, V)
    else:
        new_precision = precision

    assert isinstance(new_precision, Precision), (precision, new_precision)
    return new_precision

def sigmoid_mha(Q, K, V, sm_scale='default', bias='default', causal=True, precision=None):
    '''
    Compute the (fused) multi-head attention (MHA) with sigmoid scaling according to
        $O = sigmoid(mask(QK^T \cdot c + b))$ (up to broadcasting)

    Parameters
    ----------
    Q : tensor
        Query tensor of shape (B, T, H, D) where T is the sequence length.
    K : tensor
        Key tensor of shape (B, T, H, D).
    V : tensor
        Value tensor of shape (B, T, H, D).
    sm_scale : float, str, or None, optional
        Scale factor for the attention logit matrix. The default value ('default') corresponds
        to 1/sqrt(D). It can also be a specific float value or None if no scaling is desired.
    bias : float, str, or None, optional
        Bias term for the dot product. The default value ('default') corresponds to log(T) * -5 (from the Apple paper about this).
        It can also be a specific float value or None if no bias term is applied.
    causal : bool, optional
        Whether to apply a causal mask to the attention logit matrix. Defaults to True.
    precision : Precision enum, optional
        The precision of the operation. Defaults to Precision.TF32_ROUND, as lower precisions
        can be too numerically unstable for meaningful second order gradients.
        See the README.md for further details.

    Returns
    -------
    tensor
        The output tensor after applying the multi-head attention operation.
    '''
    bias, sm_scale = get_default_bias_sm_scale(bias, sm_scale, Q)
    precision = get_precision(Q, K, V, precision)

    return _sigmoid_mha(Q, K, V, sm_scale, bias, causal, precision)

def softmax_mha(Q, K, V, sm_scale='default', causal=True, precision=None):
    '''
    Compute the (fused) multi-head attention (MHA) with softmax scaling according to
        $O = softmax(\mathrm{mask}(QK^T \cdot c))$ (up to broadcasting)

    Parameters
    ----------
    Q : tensor
        Query tensor of shape (B, T, H, D) where T is the sequence length.
    K : tensor
        Key tensor of shape (B, T, H, D).
    V : tensor
        Value tensor of shape (B, T, H, D).
    sm_scale : float, str, or None, optional
        Scale factor for the attention logit matrix. The default value ('default') corresponds
        to 1/sqrt(D). It can also be a specific float value or None if no scaling is desired.
    causal : bool, optional
        Whether to apply a causal mask to the attention logit matrix. Defaults to True.
    precision : Precision enum, optional
        The precision of the operation. Defaults to Precision.TF32_ROUND, as lower precisions
        can be too numerically unstable for meaningful second order gradients.
        See the README.md for further details.

    Returns
    -------
    tensor
        The output tensor after applying the multi-head attention operation.
    '''
    _, sm_scale = get_default_bias_sm_scale(None, sm_scale, Q)
    precision = get_precision(Q, K, V, precision)

    return _softmax_mha(Q, K, V, sm_scale, causal, precision)

def naive_sigmoid_mha(Q, K, V, sm_scale='default', bias='default', causal=True,
                      precision=None):
    '''
    Compute the multi-head attention (MHA) with sigmoid scaling according to
        $O = sigmoid(mask(QK^T \cdot c + b))$ (up to broadcasting)

    Parameters
    ----------
    Q : tensor
        Query tensor of shape (B, T, H, D) where T is the sequence length.
    K : tensor
        Key tensor of shape (B, T, H, D).
    V : tensor
        Value tensor of shape (B, T, H, D).
    sm_scale : float, str, or None, optional
        Scale factor for the attention logit matrix. The default value ('default') corresponds
        to 1/sqrt(D). It can also be a specific float value or None if no scaling is desired.
    bias : float, str, or None, optional
        Bias term for the dot product. The default value ('default') corresponds to log(T) * -5.
        It can also be a specific float value or None if no bias term is applied.
    causal : bool, optional
        Whether to apply a causal mask to the attention logit matrix. Defaults to True.
    precision : Precision enum, optional
        The precision of the operation. Defaults to Precision.TF32_ROUND, as lower precisions
        can be too numerically unstable for meaningful second order gradients.
        See the README.md for further details.

    Returns
    -------
    tensor
        The output tensor after applying the multi-head attention operation.
    '''
    bias, sm_scale = get_default_bias_sm_scale(bias, sm_scale, Q)
    precision = get_precision(Q, K, V, precision)

    return _naive_sigmoid_mha(Q, K, V, sm_scale, causal, bias, precision)

def naive_softmax_mha(Q, K, V, sm_scale='default', causal=True, precision=None):
    '''
    Compute the multi-head attention (MHA) with softmax scaling according to
        $O = softmax(\mathrm{mask}(QK^T \cdot c))$ (up to broadcasting)

    Parameters
    ----------
    Q : tensor
        Query tensor of shape (B, T, H, D) where T is the sequence length.
    K : tensor
        Key tensor of shape (B, T, H, D).
    V : tensor
        Value tensor of shape (B, T, H, D).
    sm_scale : float, str, or None, optional
        Scale factor for the attention logit matrix. The default value ('default') corresponds
        to 1/sqrt(D). It can also be a specific float value or None if no scaling is desired.
    causal : bool, optional
        Whether to apply a causal mask to the attention logit matrix. Defaults to True.
    precision : Precision enum, optional
        The precision of the operation. Defaults to Precision.TF32_ROUND, as lower precisions
        can be too numerically unstable for meaningful second order gradients.
        See the README.md for further details.

    Returns
    -------
    tensor
        The output tensor after applying the multi-head attention operation.
    '''
    _, sm_scale = get_default_bias_sm_scale(None, sm_scale, Q)
    precision = get_precision(Q, K, V, precision)

    return _naive_softmax_mha(Q, K, V, sm_scale, causal, precision)

