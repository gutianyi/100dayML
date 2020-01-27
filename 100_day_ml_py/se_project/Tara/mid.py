def mid(x, y, z):
    m = z
    if y < z:
        if x < y:
            m = z  # ***BUG*** set z should be m = y
        elif x < z:
            m = y  # ***BUG*** set y should be m = x
    elif x > y:
        m = x  # ***BUG*** set x should be m = y
    elif x > z:
        m = x  # ***BUG*** set y should be m = x
    return m
