    def extract_features_zero_copy(self, processed_audio: np.ndarray) -> np.ndarray:
        """Extracción híbrida: encoder ONNX estándar + projection zero-copy."""
        
        # PASO 1: Encoder ONNX estándar (para obtener tamaño exacto)
        audio_shape = np.array([[processed_audio.shape[1]]], dtype=np.int64)
        encoder_inputs = {
            'audio': processed_audio,
            'audio_shape': audio_shape
        }
        encoder_outputs = self.encoder_session.run(None, encoder_inputs)
        encoder_features = encoder_outputs[0]  # [1, 768, time_frames] con tamaño exacto
        
        # PASO 2: Projection con zero-copy (ahora conocemos el tamaño exacto)
        actual_time_frames = encoder_features.shape[2]  # Tamaño real del encoder
        
        self.projection_io_binding.bind_input(
            name=self.projection_session.get_inputs()[0].name,
            device_type='cpu',
            device_id=0,
            element_type=np.float32,
            shape=encoder_features.shape,
            buffer_ptr=encoder_features.ctypes.data
        )
        
        # Output buffer para projection - [1, 256, time_frames] con tamaño exacto
        projection_output_shape = [1, 256, actual_time_frames]
        projection_output = np.empty(projection_output_shape, dtype=np.float32)
        self.projection_io_binding.bind_output(
            name=self.projection_session.get_outputs()[0].name,
            device_type='cpu',
            device_id=0,
            element_type=np.float32,
            shape=projection_output_shape,
            buffer_ptr=projection_output.ctypes.data
        )
        
        # Run projection con zero-copy
        self.projection_session.run_with_iobinding(self.projection_io_binding)
        
        return projection_output