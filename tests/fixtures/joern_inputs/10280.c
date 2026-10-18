void CIRCSock::ConnectionRefused() {
    DEBUG(GetSockName() << " == ConnectionRefused()");
    if (!m_pNetwork->GetUser()->IsBeingDeleted()) {
        m_pNetwork->PutStatus(t_s("Connection Refused.  Reconnecting..."));
    }
    m_pNetwork->ClearRawBuffer();
    m_pNetwork->ClearMotdBuffer();
}