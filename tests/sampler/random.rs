use optimizer::parameter::{CategoricalParam, FloatParam, IntParam, Parameter};
use optimizer::sampler::random::RandomSampler;
use optimizer::{Direction, Error, Study};

#[test]
fn test_random_sampler_uniform_float_distribution() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));

    let n_samples = 1000;
    let mut samples = Vec::with_capacity(n_samples);

    let x_param = FloatParam::new(0.0, 1.0);

    study
        .optimize(n_samples, |trial| {
            let x = x_param.suggest(trial)?;
            samples.push(x);
            Ok::<_, Error>(x)
        })
        .unwrap();

    // All samples should be in range
    for &s in &samples {
        assert!((0.0..=1.0).contains(&s), "sample {s} out of range [0, 1]");
    }

    // Check distribution is roughly uniform by looking at quartiles
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let q1 = samples[n_samples / 4];
    let q2 = samples[n_samples / 2];
    let q3 = samples[3 * n_samples / 4];

    assert!((q1 - 0.25).abs() < 0.1, "Q1 {q1} should be close to 0.25");
    assert!(
        (q2 - 0.5).abs() < 0.1,
        "Q2 (median) {q2} should be close to 0.5"
    );
    assert!((q3 - 0.75).abs() < 0.1, "Q3 {q3} should be close to 0.75");
}

#[test]
fn test_random_sampler_uniform_int_distribution() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(123));

    let n_samples = 5000;
    let mut counts = [0u32; 10]; // counts for values 1-10

    let n_param = IntParam::new(1, 10);

    study
        .optimize(n_samples, |trial| {
            let n = n_param.suggest(trial)?;
            assert!((1..=10).contains(&n), "sample {n} out of range [1, 10]");
            counts[(n - 1) as usize] += 1;
            Ok::<_, Error>(n as f64)
        })
        .unwrap();

    let expected = n_samples as f64 / 10.0;
    for (i, &count) in counts.iter().enumerate() {
        let diff = (count as f64 - expected).abs() / expected;
        assert!(
            diff < 0.2,
            "value {} appeared {} times, expected ~{}, diff = {:.1}%",
            i + 1,
            count,
            expected,
            diff * 100.0
        );
    }
}

#[test]
fn test_random_sampler_uniform_categorical_distribution() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(456));

    let n_samples = 2000;
    let mut counts = [0u32; 4];
    let choices = ["a", "b", "c", "d"];

    let cat_param = CategoricalParam::new(choices.to_vec());

    study
        .optimize(n_samples, |trial| {
            let choice = cat_param.suggest(trial)?;
            let idx = choices.iter().position(|&c| c == choice).unwrap();
            counts[idx] += 1;
            Ok::<_, Error>(idx as f64)
        })
        .unwrap();

    let expected = n_samples as f64 / 4.0;
    for (i, &count) in counts.iter().enumerate() {
        let diff = (count as f64 - expected).abs() / expected;
        assert!(
            diff < 0.15,
            "category {} appeared {} times, expected ~{}, diff = {:.1}%",
            i,
            count,
            expected,
            diff * 100.0
        );
    }
}

#[test]
fn test_random_sampler_reproducibility() {
    let study1: Study<f64> =
        Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(999));
    let study2: Study<f64> =
        Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(999));

    let mut values1 = Vec::new();
    let mut values2 = Vec::new();

    let x_param1 = FloatParam::new(0.0, 100.0);
    let x_param2 = FloatParam::new(0.0, 100.0);

    study1
        .optimize(100, |trial| {
            let x = x_param1.suggest(trial)?;
            values1.push(x);
            Ok::<_, Error>(x)
        })
        .unwrap();

    study2
        .optimize(100, |trial| {
            let x = x_param2.suggest(trial)?;
            values2.push(x);
            Ok::<_, Error>(x)
        })
        .unwrap();

    for (i, (v1, v2)) in values1.iter().zip(values2.iter()).enumerate() {
        assert_eq!(
            v1, v2,
            "values at trial {i} should be identical with same seed: {v1} vs {v2}"
        );
    }
}
