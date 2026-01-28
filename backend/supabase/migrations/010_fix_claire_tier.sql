-- Fix Claire's tier - she should be unlimited
-- The API key is linked to usr_73b0fcaeab13 (old claire agent)
UPDATE agents 
SET tier = 'unlimited' 
WHERE user_id = 'usr_73b0fcaeab13'
  AND agent_id = 'claire';
