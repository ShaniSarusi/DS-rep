def close_connection(co):
    csr = co.cursor()
    csr.close()
    del csr
    co.close()