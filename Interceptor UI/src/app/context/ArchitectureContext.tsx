import React, { createContext, useContext, useState, useCallback } from 'react';

interface ArchitectureState {
  currentPage: string;
  activeLayer: number;
  activeModels: string[];
  dataFlow: DataFlow[];
  processingStage: string;
}

interface DataFlow {
  id: string;
  from: string;
  to: string;
  data: string;
  progress: number;
}

interface ArchitectureContextType {
  state: ArchitectureState;
  setCurrentPage: (page: string) => void;
  setActiveLayer: (layer: number) => void;
  activateModel: (model: string) => void;
  deactivateModel: (model: string) => void;
  startDataFlow: (from: string, to: string, data: string) => void;
  setProcessingStage: (stage: string) => void;
  resetFlow: () => void;
}

const ArchitectureContext = createContext<ArchitectureContextType | undefined>(undefined);

export const ArchitectureProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, setState] = useState<ArchitectureState>({
    currentPage: 'home',
    activeLayer: 0,
    activeModels: [],
    dataFlow: [],
    processingStage: 'idle',
  });

  const setCurrentPage = useCallback((page: string) => {
    setState(prev => ({ ...prev, currentPage: page, processingStage: 'idle' }));
  }, []);

  const setActiveLayer = useCallback((layer: number) => {
    setState(prev => ({ ...prev, activeLayer: layer }));
  }, []);

  const activateModel = useCallback((model: string) => {
    setState(prev => ({
      ...prev,
      activeModels: [...prev.activeModels, model].filter((v, i, a) => a.indexOf(v) === i),
    }));
  }, []);

  const deactivateModel = useCallback((model: string) => {
    setState(prev => ({
      ...prev,
      activeModels: prev.activeModels.filter(m => m !== model),
    }));
  }, []);

  const startDataFlow = useCallback((from: string, to: string, data: string) => {
    const flowId = `${from}-${to}-${Date.now()}`;
    setState(prev => ({
      ...prev,
      dataFlow: [...prev.dataFlow, { id: flowId, from, to, data, progress: 0 }],
    }));
  }, []);

  const setProcessingStage = useCallback((stage: string) => {
    setState(prev => ({ ...prev, processingStage: stage }));
  }, []);

  const resetFlow = useCallback(() => {
    setState(prev => ({
      ...prev,
      activeModels: [],
      dataFlow: [],
      processingStage: 'idle',
      activeLayer: 0,
    }));
  }, []);

  return (
    <ArchitectureContext.Provider
      value={{
        state,
        setCurrentPage,
        setActiveLayer,
        activateModel,
        deactivateModel,
        startDataFlow,
        setProcessingStage,
        resetFlow,
      }}
    >
      {children}
    </ArchitectureContext.Provider>
  );
};

export const useArchitecture = () => {
  const context = useContext(ArchitectureContext);
  if (!context) {
    throw new Error('useArchitecture must be used within ArchitectureProvider');
  }
  return context;
};
