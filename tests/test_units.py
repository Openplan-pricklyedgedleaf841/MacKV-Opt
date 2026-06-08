from mackv_opt.units import format_bytes, parse_context, parse_size


def test_parse_size_accepts_decimal_and_binary_units():
    assert parse_size("20GB") == 20_000_000_000
    assert parse_size("16GiB") == 17_179_869_184
    assert parse_size("512m") == 512_000_000


def test_parse_context_accepts_k_suffix():
    assert parse_context("64k") == 65_536
    assert parse_context("8192") == 8192


def test_format_bytes_uses_gib_for_human_readable_memory():
    assert format_bytes(17_179_869_184) == "16.00 GiB"
