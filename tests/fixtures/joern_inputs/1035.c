uint32_t BinaryProtocolWriter::writeFieldStop() {
  return writeByte((int8_t)TType::T_STOP);
}