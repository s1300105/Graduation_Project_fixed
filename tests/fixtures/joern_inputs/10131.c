f_matcharg(typval_T *argvars UNUSED, typval_T *rettv)
{
    if (rettv_list_alloc(rettv) == OK)
    {
#ifdef FEAT_SEARCH_EXTRA
	int	    id = (int)tv_get_number(&argvars[0]);
	matchitem_T *m;

	if (id >= 1 && id <= 3)
	{
	    if ((m = (matchitem_T *)get_match(curwin, id)) != NULL)
	    {
		list_append_string(rettv->vval.v_list,
						syn_id2name(m->hlg_id), -1);
		list_append_string(rettv->vval.v_list, m->pattern, -1);
	    }
	    else
	    {
		list_append_string(rettv->vval.v_list, NULL, -1);
		list_append_string(rettv->vval.v_list, NULL, -1);
	    }
	}
#endif
    }
}