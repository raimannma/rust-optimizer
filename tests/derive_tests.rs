use optimizer::Trial;
use optimizer::parameter::{EnumParam, Parameter};
use optimizer_derive::Categorical;

#[derive(Clone, Debug, PartialEq, Categorical)]
enum Color {
    Red,
    Green,
    Blue,
}

#[derive(Clone, Debug, PartialEq, Categorical)]
enum SingleVariant {
    Only,
}

#[test]
fn derive_categorical_n_choices() {
    use optimizer::parameter::Categorical;
    assert_eq!(Color::N_CHOICES, 3);
    assert_eq!(SingleVariant::N_CHOICES, 1);
}

#[test]
fn derive_categorical_roundtrip() {
    use optimizer::parameter::Categorical;
    for i in 0..Color::N_CHOICES {
        let val = Color::from_index(i);
        assert_eq!(val.to_index(), i);
    }
}

#[test]
fn derive_categorical_values() {
    use optimizer::parameter::Categorical;
    assert_eq!(Color::from_index(0), Color::Red);
    assert_eq!(Color::from_index(1), Color::Green);
    assert_eq!(Color::from_index(2), Color::Blue);
    assert_eq!(Color::Red.to_index(), 0);
    assert_eq!(Color::Green.to_index(), 1);
    assert_eq!(Color::Blue.to_index(), 2);
}

#[test]
fn derive_categorical_with_enum_param() {
    let mut trial = Trial::new(0);
    let param = EnumParam::<Color>::new();
    let color = param.suggest(&mut trial).unwrap();
    assert!([Color::Red, Color::Green, Color::Blue].contains(&color));

    // Cached (same param id)
    let color2 = param.suggest(&mut trial).unwrap();
    assert_eq!(color, color2);
}

#[test]
fn derive_categorical_suggest_via_trial() {
    let mut trial = Trial::new(0);
    let color = trial.suggest_param(&EnumParam::<Color>::new()).unwrap();
    assert!([Color::Red, Color::Green, Color::Blue].contains(&color));
}
