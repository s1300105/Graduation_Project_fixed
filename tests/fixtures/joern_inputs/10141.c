uint32_t BinaryProtocolWriter::serializedSizeString(
    folly::StringPiece str) const {
  return serializedSizeBinary(str);
}