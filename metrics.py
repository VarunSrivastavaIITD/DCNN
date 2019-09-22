import numpy as np


def convert_wave_to_positions(arr):
    assert (
        len(arr) == np.count_nonzero(arr == 0) + np.count_nonzero(arr == 1))
    return np.nonzero(arr == 1)


def naylor_metrics(ref_signal, est_signal):
    # Settings
    # TODO: precise values to be decided later

    assert (np.squeeze(ref_signal).ndim == 1)
    assert (np.squeeze(est_signal).ndim == 1)

    ref_signal = np.squeeze(ref_signal)
    est_signal = np.squeeze(est_signal)

    min_f0 = 50
    max_f0 = 500
    min_glottal_cycle = 1 / max_f0
    max_glottal_cycle = 1 / min_f0

    nHit = 0
    nMiss = 0
    nFalse = 0
    nCycles = 0
    highNumCycles = 100000
    estimation_distance = np.full(highNumCycles, np.nan)

    ref_fwdiffs = np.diff(ref_signal)[1:]
    ref_bwdiffs = np.diff(ref_signal)[:-1]

    for i in range(len(ref_fwdiffs)):

        # m in original file
        ref_cur_sample = ref_signal[i + 1]
        ref_dist_fw = ref_fwdiffs[i]
        ref_dist_bw = ref_bwdiffs[i]

        # Condition to check for valid larynx cycle
        # TODO: Check parity of differences, neg peak <-> gci, pos peak <-> goi
        # TODO: Check applicability of strict inequality
        dist_in_allowed_range = min_glottal_cycle <= ref_dist_fw <= max_glottal_cycle and \
            min_glottal_cycle <= ref_dist_bw <= max_glottal_cycle
        if dist_in_allowed_range:

            cycle_start = ref_cur_sample - ref_dist_bw / 2
            cycle_stop = ref_cur_sample + ref_dist_fw / 2

            est_GCIs_in_cycle = est_signal[np.logical_and(
                est_signal > cycle_start, est_signal < cycle_stop)]
            n_est_in_cycle = np.count_nonzero(est_GCIs_in_cycle)

            nCycles += 1

            if n_est_in_cycle == 1:
                nHit += 1
                estimation_distance[nHit] = est_GCIs_in_cycle[0] - \
                    ref_cur_sample
            elif n_est_in_cycle < 1:
                nMiss += 1
            else:
                nFalse += 1

    estimation_distance = estimation_distance[np.invert(
        np.isnan(estimation_distance))]

    identification_rate = nHit / nCycles
    miss_rate = nMiss / nCycles
    false_alarm_rate = nFalse / nCycles
    identification_accuracy = 0 if np.size(
        estimation_distance) == 0 else np.std(estimation_distance)

    return {
        'identification_rate': identification_rate,
        'miss_rate': miss_rate,
        'false_alarm_rate': false_alarm_rate,
        'identification_accuracy': identification_accuracy
    }


def corrected_naylor_metrics(ref_signal, est_signal):
    # Settings
    # TODO: precise values to be decided later

    assert (np.squeeze(ref_signal).ndim == 1)
    assert (np.squeeze(est_signal).ndim == 1)

    ref_signal = np.squeeze(ref_signal)
    est_signal = np.squeeze(est_signal)

    min_f0 = 50
    max_f0 = 500
    min_glottal_cycle = 1 / max_f0
    max_glottal_cycle = 1 / min_f0

    nHit = 0
    nMiss = 0
    nFalse = 0
    nCycles = 0
    highNumCycles = 100000
    estimation_distance = np.full(highNumCycles, np.nan)

    ref_fwdiffs = np.zeros_like(ref_signal)
    ref_bwdiffs = np.zeros_like(ref_signal)

    ref_fwdiffs[:-1] = np.diff(ref_signal)
    ref_fwdiffs[-1] = max_glottal_cycle
    ref_bwdiffs[1:] = np.diff(ref_signal)
    ref_bwdiffs[0] = max_glottal_cycle

    for i in range(len(ref_fwdiffs)):

        # m in original file
        ref_cur_sample = ref_signal[i]
        ref_dist_fw = ref_fwdiffs[i]
        ref_dist_bw = ref_bwdiffs[i]

        # Condition to check for valid larynx cycle
        # TODO: Check parity of differences, neg peak <-> gci, pos peak <-> goi
        # TODO: Check applicability of strict inequality
        dist_in_allowed_range = min_glottal_cycle <= ref_dist_fw <= max_glottal_cycle and \
            min_glottal_cycle <= ref_dist_bw <= max_glottal_cycle
        if dist_in_allowed_range:

            cycle_start = ref_cur_sample - ref_dist_bw / 2
            cycle_stop = ref_cur_sample + ref_dist_fw / 2

            est_GCIs_in_cycle = est_signal[np.logical_and(
                est_signal > cycle_start, est_signal < cycle_stop)]
            n_est_in_cycle = np.count_nonzero(est_GCIs_in_cycle)

            nCycles += 1

            if n_est_in_cycle == 1:
                nHit += 1
                estimation_distance[nHit] = est_GCIs_in_cycle[0] - \
                    ref_cur_sample
            elif n_est_in_cycle < 1:
                nMiss += 1
            else:
                nFalse += 1

    estimation_distance = estimation_distance[np.invert(
        np.isnan(estimation_distance))]

    identification_rate = nHit / nCycles
    miss_rate = nMiss / nCycles
    false_alarm_rate = nFalse / nCycles
    identification_accuracy = 0 if np.size(
        estimation_distance) == 0 else np.std(estimation_distance)

    return {
        'identification_rate': identification_rate,
        'miss_rate': miss_rate,
        'false_alarm_rate': false_alarm_rate,
        'identification_accuracy': identification_accuracy
    }


def adjusted_naylor_metrics(ref_signal, est_signal):
    # Settings
    # TODO: precise values to be decided later

    assert (np.squeeze(ref_signal).ndim == 1)
    assert (np.squeeze(est_signal).ndim == 1)

    ref_signal = np.squeeze(ref_signal)
    est_signal = np.squeeze(est_signal)

    min_f0 = 50
    max_f0 = 500
    min_glottal_cycle = 1 / max_f0
    max_glottal_cycle = 1 / min_f0

    nHit = 0
    nMiss = 0
    nFalse = 0
    nCycles = 0
    highNumCycles = 100000
    estimation_distance = np.full(highNumCycles, np.nan)

    ref_fwdiffs = np.zeros_like(ref_signal)
    ref_bwdiffs = np.zeros_like(ref_signal)

    ref_fwdiffs[:-1] = np.diff(ref_signal)
    ref_bwdiffs[1:] = np.diff(ref_signal)
    ref_fwdiffs[-1] = ref_bwdiffs[-1]
    ref_bwdiffs[0] = ref_fwdiffs[0]

    for i in range(len(ref_fwdiffs)):

        # m in original file
        ref_cur_sample = ref_signal[i]
        ref_dist_fw = ref_fwdiffs[i]
        ref_dist_bw = ref_bwdiffs[i]

        # Condition to check for valid larynx cycle
        # TODO: Check parity of differences, neg peak <-> gci, pos peak <-> goi
        # TODO: Check applicability of strict inequality

        bw_allowed_range = min_glottal_cycle <= ref_dist_bw <= max_glottal_cycle
        fw_allowed_range = min_glottal_cycle <= ref_dist_fw <= max_glottal_cycle

        if bw_allowed_range and ref_dist_fw > max_glottal_cycle:
            ref_dist_fw = ref_dist_bw
        elif fw_allowed_range and ref_dist_bw > max_glottal_cycle:
            ref_dist_bw = ref_dist_fw

        dist_in_allowed_range = fw_allowed_range and bw_allowed_range
        if dist_in_allowed_range:

            cycle_start = ref_cur_sample - ref_dist_bw / 2
            cycle_stop = ref_cur_sample + ref_dist_fw / 2

            est_GCIs_in_cycle = est_signal[np.logical_and(
                est_signal > cycle_start, est_signal < cycle_stop)]
            n_est_in_cycle = np.count_nonzero(est_GCIs_in_cycle)

            nCycles += 1

            if n_est_in_cycle == 1:
                nHit += 1
                estimation_distance[nHit] = est_GCIs_in_cycle[0] - \
                    ref_cur_sample
            elif n_est_in_cycle < 1:
                nMiss += 1
            else:
                nFalse += 1

    estimation_distance = estimation_distance[np.invert(
        np.isnan(estimation_distance))]

    identification_rate = nHit / nCycles
    miss_rate = nMiss / nCycles
    false_alarm_rate = nFalse / nCycles
    identification_accuracy = 0 if np.size(
        estimation_distance) == 0 else np.std(estimation_distance)

    return {
        'identification_rate': identification_rate,
        'miss_rate': miss_rate,
        'false_alarm_rate': false_alarm_rate,
        'identification_accuracy': identification_accuracy
    }


def frames_to_time_clusters(frames, sample_indices):
    indices = list(set(sample_indices)).sort()
    frames = np.squeeze(frames)
    assert (frames.ndim == 2)
    linear_frames = frames.ravel()
    result = [linear_frames[sample_indices == i] for i in indices]
    return result, indices


def main():
    ref = np.array([0.0975, 0.1, 0.11, 0.125, 0.127, 0.140, 0.145, 0.163])
    est = np.array([0.099, 0.114, 0.128, 0.132, 0.146])

    val = naylor_metrics(ref, est)
    print(val)


if __name__ == "__main__":
    main()
