void CClient::ClearServerDependentCaps() {
    if (HasCapNotify() && !m_ssServerDependentCaps.empty()) {
        CString sCaps = CString(" ").Join(m_ssServerDependentCaps.begin(),
                                          m_ssServerDependentCaps.end());
        PutClient(":irc.znc.in CAP " + GetNick() + " DEL :" + sCaps);

        for (const CString& sCap : m_ssServerDependentCaps) {
            const auto& it = m_mCoreCaps.find(sCap);
            if (m_mCoreCaps.end() != it) {
                const auto& handler = std::get<1>(it->second);
                handler(false);
            }
        }
    }

    m_ssServerDependentCaps.clear();
}