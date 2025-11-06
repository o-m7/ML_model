import React, { useState, useEffect } from 'react';
import { supabase, LiveSignal } from './supabaseClient';
import { SignalCard } from './SignalCard';

export const SignalDashboard: React.FC = () => {
  const [signals, setSignals] = useState<LiveSignal[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'active' | 'high'>('active');

  useEffect(() => {
    loadSignals();
    
    // Subscribe to real-time updates
    const subscription = supabase
      .channel('live_signals_channel')
      .on('postgres_changes', {
        event: 'INSERT',
        schema: 'public',
        table: 'live_signals'
      }, (payload) => {
        console.log('New signal:', payload.new);
        setSignals(prev => [payload.new as LiveSignal, ...prev]);
        
        // Show notification for high-quality signals
        if ((payload.new as any).confidence > 0.5) {
          showNotification(payload.new as LiveSignal);
        }
      })
      .subscribe();

    return () => {
      subscription.unsubscribe();
    };
  }, []);

  const loadSignals = async () => {
    setLoading(true);
    
    let query = supabase
      .from('live_signals')
      .select('*')
      .order('timestamp', { ascending: false })
      .limit(50);
    
    if (filter === 'active') {
      query = query.eq('status', 'active');
    }
    
    const { data, error } = await query;
    
    if (error) {
      console.error('Error loading signals:', error);
    } else {
      setSignals(data || []);
    }
    
    setLoading(false);
  };

  const showNotification = (signal: LiveSignal) => {
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification('🎯 New Trading Signal!', {
        body: `${signal.symbol} ${signal.timeframe}: ${signal.signal_type.toUpperCase()} - ${(signal.confidence * 100).toFixed(1)}%`,
        icon: '/logo.png'
      });
    }
  };

  const requestNotificationPermission = () => {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  };

  const groupedSignals = signals.reduce((acc, signal) => {
    const key = `${signal.symbol}_${signal.timeframe}`;
    if (!acc[key] || new Date(signal.timestamp) > new Date(acc[key].timestamp)) {
      acc[key] = signal;
    }
    return acc;
  }, {} as Record<string, LiveSignal>);

  const latestSignals = Object.values(groupedSignals);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="signal-dashboard">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">Live Trading Signals</h2>
        
        <div className="flex gap-2">
          <button
            onClick={requestNotificationPermission}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            🔔 Enable Alerts
          </button>
          
          <button
            onClick={loadSignals}
            className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
          >
            🔄 Refresh
          </button>
        </div>
      </div>

      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setFilter('all')}
          className={`px-4 py-2 rounded ${filter === 'all' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
        >
          All ({signals.length})
        </button>
        <button
          onClick={() => setFilter('active')}
          className={`px-4 py-2 rounded ${filter === 'active' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
        >
          Active ({signals.filter(s => s.status === 'active').length})
        </button>
      </div>

      {latestSignals.length === 0 ? (
        <div className="text-center p-8 bg-gray-100 rounded-lg">
          <p className="text-gray-600">No signals available. Run the live trading engine to generate signals.</p>
          <code className="text-sm bg-gray-200 px-2 py-1 rounded mt-2 inline-block">
            python3 live_trading_engine.py
          </code>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {latestSignals.map(signal => (
            <SignalCard key={signal.id} signal={signal} />
          ))}
        </div>
      )}
    </div>
  );
};

