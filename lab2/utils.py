import numpy as np
import cv2
import matplotlib.pyplot as plt


def power_method(
    cov: np.array, max_iter: int = 50, tol: float = 1e-8
) -> tuple[float, np.array]:
    """Степенной метод для нахождения собственного значения максималььного по модулю
    и соответствующего ему собственного вектора симметрической матрицы

    Args:
        cov (np.array): матрица (ковариации)
        max_iter (int, optional): кол-во итераций алгоритма. Defaults to 50.
        tol (float, optional): пороговая погрешность

    Raises:
        ValueError: в случае невалидного max_iter
        ValueError: в случае, если передаваемый тензор - не матрица
        ValueError: если матрица несимметрическая

    Returns:
        tuple[float, np.array]: собственное значение максимального модуля и соот. ему вектор
    """
    if max_iter <= 0:
        raise ValueError("Invalid max_iter value")
    if cov.ndim != 2:
        raise ValueError("2-dimensional matrices allowed only")
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("Square matrices allowed only")

    n = cov.shape[0]
    eigenvec = np.random.uniform(-1, 1, n)
    eigenvec /= np.linalg.norm(eigenvec)
    prev = eigenvec
    eigenvec = cov @ eigenvec

    for _ in range(max_iter - 1):
        if np.linalg.norm(eigenvec - prev) < tol:
            break
        prev = eigenvec
        eigenvec = cov @ eigenvec
        eigenvec /= np.linalg.norm(eigenvec)

    lam = (eigenvec.T @ cov @ eigenvec) / (eigenvec.T @ eigenvec)

    return (lam, eigenvec)


def svd_via_power_method(
    X: np.array,
    max_iter: int = 50,
    enable_below_zero_sigma_warning: bool = True,
) -> tuple[np.array, np.array, np.array]:
    """Сингулярное разложение на основе степенного метода

    Args:
        X (np.array): матрица данных
        max_iter (int, optional): кол-во итераций. Defaults to 50.
        enable_below_zero_sigma_warning (bool, optional): предупреждать ли
        об отрицательных значениях. Defaults to True.

    Raises:
        ValueError: в случае невалидного max_iter
        ValueError: если передаваемый тензор - не матрица

    Returns:
        tuple[np.array, np.array, np.array]: U, S, V^T, где:
        U - матрица левых сингулярных векторов,
        S - массив сингулярных чисел,
        V - матрица правых сингулярных векторов
    """
    if max_iter <= 0:
        raise ValueError("Invalid max_iter value")
    if X.ndim != 2:
        raise ValueError("2-dimensional matrices allowed only")

    cov = X.T @ X

    U = []
    sigmas = []
    V = []

    rank = np.linalg.matrix_rank(X)

    for k in range(rank):
        lam, v = power_method(cov, max_iter)
        v = v.reshape(X.shape[1], 1)

        if lam < 0:
            if enable_below_zero_sigma_warning:
                print(
                    f"Below zero singular value detected at iteration {k}: {lam:.3e}"
                )
            sigma = np.sqrt(-lam)
        else:
            sigma = np.sqrt(lam)
        sigmas.append(sigma)
        V.append(v.T)
        U.append(X @ v / sigma)

        cov = cov - (v @ v.T) * lam
    return (
        np.concatenate(U, axis=1),
        np.array(sigmas),
        np.concatenate(V, axis=0),
    )


def trunc_recon(U, S, Vt, r=300) -> np.array:
    r = min([r, U.shape[1], S.shape[0], Vt.shape[0]])
    return U[:, :r] @ np.diag(S)[:r, :r] @ Vt[:r, :]


def jacobi_eigenvalue(
    A: np.array, tol: float = 1e-6, max_iter: int = 1000
) -> tuple[np.array, np.array, np.array]:
    """Найти собственные векторы и значения симметрической матрицы симметрической матрицы
    методом плоских вращений Якоби

    Args:
        A (np.array): симметрическая матрица
        tol (float, optional): порог ошибки нуля, после которого
        алгоритм завершается. Defaults to 1e-6.
        max_iter (int, optional): максимальное кол-во операций. Defaults to 1000.

    Raises:
        ValueError: в случае невалидного max_iter
        ValueError: в случае, если переданный тензор - не матрица
        ValueError: если переданная матрица неквадратная

    Returns:
        tuple[np.array, np.array, np.array]: _description_
    """
    if max_iter <= 0:
        raise ValueError("Invalid max_iter value")
    if len(A.shape) != 2:
        raise ValueError("Only matrices allowed!")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Only square matrices allowed!")

    n = A.shape[0]
    D = A.copy()

    error_history = []
    err = np.linalg.norm(np.triu(A, 1))
    error_history.append(err)

    eigenvecs = np.eye(n)

    for i in range(max_iter):

        p, q = np.unravel_index(
            np.argmax(np.abs(D - np.diag(np.diag(D)))), A.shape
        )
        if abs(D[p, q]) < tol:
            break

        # Вычисляем угол вращения
        if D[p, p] == D[q, q]:
            theta = np.pi / 4
        else:
            theta = 0.5 * np.arctan(2 * D[p, q] / (D[p, p] - D[q, q]))

        # Формируем матрицу вращения
        J = np.eye(n)
        c = np.cos(theta)
        s = np.sin(theta)
        J[p, p] = c
        J[q, q] = c
        J[p, q] = -s
        J[q, p] = s

        # Применяем вращение к матрице D
        D = J.T @ D @ J
        err = np.linalg.norm(np.triu(D, 1))
        error_history.append(err)

        eigenvecs = eigenvecs @ J

    D = np.diag(D)

    return np.sort(D)[::-1], eigenvecs, error_history
