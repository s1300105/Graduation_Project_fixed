void CZNC::DeleteUsers() {
    for (const auto& it : m_msUsers) {
        it.second->SetBeingDeleted(true);
        delete it.second;
    }

    m_msUsers.clear();
    DisableConnectQueue();
}