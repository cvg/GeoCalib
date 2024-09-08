"""Implementation of manifolds."""

import logging

import torch

logger = logging.getLogger(__name__)


class EuclideanManifold:
    """Simple euclidean manifold."""

    @staticmethod
    def J_plus(x: torch.Tensor) -> torch.Tensor:
        """Plus operator Jacobian."""
        return torch.eye(x.shape[-1]).to(x)

    @staticmethod
    def plus(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Plus operator."""
        return x + delta


class SphericalManifold:
    """Implementation of the spherical manifold.

    Following the derivation from 'Integrating Generic Sensor Fusion Algorithms with Sound State
    Representations through Encapsulation of Manifolds' by Hertzberg et al. (B.2, p. 25).

    Householder transformation following Algorithm 5.1.1 (p. 210) from 'Matrix Computations' by
    Golub et al.
    """

    @staticmethod
    def householder_vector(x: torch.Tensor) -> torch.Tensor:
        """Return the Householder vector and beta.

        Algorithm 5.1.1 (p. 210) from 'Matrix Computations' by Golub et al. (Johns Hopkins Studies
        in Mathematical Sciences) but using the nth element of the input vector as pivot instead of
        first.

        This computes the vector v with v(n) = 1 and beta such that H = I - beta * v * v^T is
        orthogonal and H * x = ||x||_2 * e_n.

        Args:
            x (torch.Tensor): [..., n] tensor.

        Returns:
            torch.Tensor: v of shape [..., n]
            torch.Tensor: beta of shape [...]
        """
        sigma = torch.sum(x[..., :-1] ** 2, -1)
        xpiv = x[..., -1]
        norm = torch.norm(x, dim=-1)
        if torch.any(sigma < 1e-7):
            sigma = torch.where(sigma < 1e-7, sigma + 1e-7, sigma)
            logger.warning("sigma < 1e-7")

        vpiv = torch.where(xpiv < 0, xpiv - norm, -sigma / (xpiv + norm))
        beta = 2 * vpiv**2 / (sigma + vpiv**2)
        v = torch.cat([x[..., :-1] / vpiv[..., None], torch.ones_like(vpiv)[..., None]], -1)
        return v, beta

    @staticmethod
    def apply_householder(y: torch.Tensor, v: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Apply Householder transformation.

        Args:
            y (torch.Tensor): Vector to transform of shape [..., n].
            v (torch.Tensor): Householder vector of shape [..., n].
            beta (torch.Tensor): Householder beta of shape [...].

        Returns:
            torch.Tensor: Transformed vector of shape [..., n].
        """
        return y - v * (beta * torch.einsum("...i,...i->...", v, y))[..., None]

    @classmethod
    def J_plus(cls, x: torch.Tensor) -> torch.Tensor:
        """Plus operator Jacobian."""
        v, beta = cls.householder_vector(x)
        H = -torch.einsum("..., ...k, ...l->...kl", beta, v, v)
        H = H + torch.eye(H.shape[-1]).to(H)
        return H[..., :-1]  # J

    @classmethod
    def plus(cls, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Plus operator.

        Equation 109 (p. 25) from 'Integrating Generic Sensor Fusion Algorithms with Sound State
        Representations through Encapsulation of Manifolds' by Hertzberg et al. but using the nth
        element of the input vector as pivot instead of first.

        Args:
            x: point on the manifold
            delta: tangent vector
        """
        eps = 1e-7
        # keep norm is not equal to 1
        nx = torch.norm(x, dim=-1, keepdim=True)
        nd = torch.norm(delta, dim=-1, keepdim=True)

        # make sure we don't divide by zero in backward as torch.where computes grad for both
        # branches
        nd_ = torch.where(nd < eps, nd + eps, nd)
        sinc = torch.where(nd < eps, nd.new_ones(nd.shape), torch.sin(nd_) / nd_)

        # cos is applied to last dim instead of first
        exp_delta = torch.cat([sinc * delta, torch.cos(nd)], -1)

        v, beta = cls.householder_vector(x)
        return nx * cls.apply_householder(exp_delta, v, beta)
