PQsetErrorContextVisibility(PGconn *conn, PGContextVisibility show_context)
{
	PGContextVisibility old;

	if (!conn)
		return PQSHOW_CONTEXT_ERRORS;
	old = conn->show_context;
	conn->show_context = show_context;
	return old;
}