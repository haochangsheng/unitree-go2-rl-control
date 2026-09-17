import argparse
import time
import numpy as np
import torch
import torch.nn as nn


OBS_LEN = 48


def rebuild_mlp(state_dict, device):
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.removeprefix('model.')] = v

    layers = []
    weight_keys = sorted([k for k in cleaned if 'weight' in k])

    for i, wk in enumerate(weight_keys):
        w = cleaned[wk]
        layers.append(nn.Linear(w.shape[1], w.shape[0]))
        if i < len(weight_keys) - 1:
            layers.append(nn.ReLU())

    net = nn.Sequential(*layers).to(device)
    net.load_state_dict(cleaned)
    return net


def load_policy(path, device):
    ckpt = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    print(f'Network structure:\n{ckpt["network_structure"]}')

    state_dict = ckpt['network_state_dict']
    net = rebuild_mlp(state_dict, device)
    net.eval()

    obs_mask = ckpt.get('actor_obs_mask')

    if obs_mask is not None:
        obs_mask = obs_mask.to(device)
        print(
            f'Actor obs mask: '
            f'{obs_mask.sum().int().item()} / {obs_mask.shape[0]} dims'
        )

    return net, obs_mask


def synchronize(device):
    if device.startswith('cuda'):
        torch.cuda.synchronize()


def benchmark(
    policy_net,
    obs_mask,
    device,
    n_warmup=1000,
    n_iterations=10000,
):
    # Fake observation.
    # Shape and dtype are the same as deployment.
    obs = np.zeros(OBS_LEN, dtype=np.float32)

    # ------------------------------------------------------------
    # Prepare input
    # ------------------------------------------------------------
    obs_t = torch.tensor(
        obs,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    if obs_mask is not None:
        obs_t = obs_t[:, obs_mask.bool()]

    print(f'\nInput shape: {tuple(obs_t.shape)}')
    print(f'Device:      {device}')
    print(f'Dtype:       {obs_t.dtype}')

    # ------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------
    print(f'\nWarm-up: {n_warmup} iterations...')

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = policy_net(obs_t)

    synchronize(device)

    # ------------------------------------------------------------
    # Benchmark pure network forward
    # ------------------------------------------------------------
    print(f'Benchmarking: {n_iterations} iterations...')

    times = np.empty(n_iterations, dtype=np.float64)

    with torch.no_grad():
        for i in range(n_iterations):
            synchronize(device)
            t0 = time.perf_counter()

            _ = policy_net(obs_t)

            synchronize(device)
            t1 = time.perf_counter()

            times[i] = (t1 - t0) * 1000.0  # ms

    print('\n========== Network Forward ==========')
    print(f'Mean:   {np.mean(times):.4f} ms')
    print(f'Median: {np.median(times):.4f} ms')
    print(f'P90:    {np.percentile(times, 90):.4f} ms')
    print(f'P95:    {np.percentile(times, 95):.4f} ms')
    print(f'P99:    {np.percentile(times, 99):.4f} ms')
    print(f'Min:    {np.min(times):.4f} ms')
    print(f'Max:    {np.max(times):.4f} ms')

    mean_ms = np.mean(times)

    if mean_ms > 0:
        print(f'\nEquivalent inference rate: {1000.0 / mean_ms:.1f} Hz')

    # ------------------------------------------------------------
    # Benchmark exactly the deployment-side input preparation
    # ------------------------------------------------------------
    print('\n========== Deployment Input + Network ==========')

    times_full = np.empty(n_iterations, dtype=np.float64)

    with torch.no_grad():
        for i in range(n_iterations):
            synchronize(device)
            t0 = time.perf_counter()

            # Same code as deployment
            obs_t = torch.tensor(
                obs,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            if obs_mask is not None:
                obs_t = obs_t[:, obs_mask.bool()]

            action = policy_net(obs_t).squeeze(0).cpu().numpy()

            synchronize(device)
            t1 = time.perf_counter()

            times_full[i] = (t1 - t0) * 1000.0

    print(f'Mean:   {np.mean(times_full):.4f} ms')
    print(f'Median: {np.median(times_full):.4f} ms')
    print(f'P95:    {np.percentile(times_full, 95):.4f} ms')
    print(f'P99:    {np.percentile(times_full, 99):.4f} ms')
    print(f'Min:    {np.min(times_full):.4f} ms')
    print(f'Max:    {np.max(times_full):.4f} ms')

    mean_full_ms = np.mean(times_full)

    if mean_full_ms > 0:
        print(
            f'\nEquivalent full inference rate: '
            f'{1000.0 / mean_full_ms:.1f} Hz'
        )

    print('\n========================================')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--policy_path',
        type=str,
        default='/workspace/policy.pt'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu'
    )

    parser.add_argument(
        '--warmup',
        type=int,
        default=1000
    )

    parser.add_argument(
        '--iterations',
        type=int,
        default=10000
    )

    args = parser.parse_args()

    print(f'Loading policy from {args.policy_path}')
    print(f'Device: {args.device}')

    policy_net, obs_mask = load_policy(
        args.policy_path,
        args.device
    )

    benchmark(
        policy_net,
        obs_mask,
        args.device,
        args.warmup,
        args.iterations
    )


if __name__ == '__main__':
    main()