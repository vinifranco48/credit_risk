-- Create loan_features table with constraints and comments
CREATE TABLE IF NOT EXISTS loan_features (
    id SERIAL PRIMARY KEY,
    loan_amnt FLOAT CHECK (loan_amnt > 0),
    term INTEGER CHECK (term IN (36, 60)),
    int_rate FLOAT CHECK (int_rate >= 0 AND int_rate <= 100),
    grade VARCHAR(1) CHECK (grade IN ('A','B','C','D','E','F','G')),
    sub_grade VARCHAR(2),
    emp_length VARCHAR(20),
    home_ownership VARCHAR(10) CHECK (home_ownership IN ('RENT','OWN','MORTGAGE','OTHER')),
    annual_inc FLOAT CHECK (annual_inc >= 0),
    verification_status VARCHAR(20),
    purpose VARCHAR(30) NOT NULL,
    addr_state VARCHAR(2),
    dti FLOAT CHECK (dti >= 0),
    inq_last_6mths INTEGER CHECK (inq_last_6mths >= 0),
    mths_since_last_delinq FLOAT,
    open_acc INTEGER CHECK (open_acc >= 0),
    revol_bal FLOAT CHECK (revol_bal >= 0),
    total_acc INTEGER CHECK (total_acc >= 0),
    initial_list_status CHAR(1) CHECK (initial_list_status IN ('w','f')),
    tot_cur_bal FLOAT CHECK (tot_cur_bal >= 0),
    mths_since_earliest_cr_line FLOAT,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create loan_predictions table with risk modeling fields
CREATE TABLE IF NOT EXISTS loan_predictions (
    id SERIAL PRIMARY KEY,
    feature_id INTEGER REFERENCES loan_features(id) ON DELETE CASCADE,
    prediction INTEGER CHECK (prediction IN (0, 1)),
    probability FLOAT CHECK (probability >= 0 AND probability <= 1),
    credit_score FLOAT CHECK (credit_score >= 300 AND credit_score <= 850),
    pd_score FLOAT CHECK (pd_score >= 0 AND pd_score <= 1),
    lgd_linear_score FLOAT CHECK (lgd_linear_score >= 0),
    lgd_logistic_score FLOAT CHECK (lgd_logistic_score >= 0 AND lgd_logistic_score <= 1),
    ead_score FLOAT CHECK (ead_score >= 0 AND ead_score <= 1),
    expected_loss FLOAT CHECK (expected_loss >= 0),
    
    prediction_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50) NOT NULL,
    minio_storage_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indices for better performance
CREATE INDEX IF NOT EXISTS idx_feature_id ON loan_predictions(feature_id);
CREATE INDEX IF NOT EXISTS idx_prediction_timestamp ON loan_predictions(prediction_timestamp);
CREATE INDEX IF NOT EXISTS idx_loan_features_status ON loan_features(status);
CREATE INDEX IF NOT EXISTS idx_loan_features_grade ON loan_features(grade);
CREATE INDEX IF NOT EXISTS idx_loan_predictions_pd ON loan_predictions(pd_score);
CREATE INDEX IF NOT EXISTS idx_loan_predictions_expected_loss ON loan_predictions(expected_loss);
CREATE INDEX IF NOT EXISTS idx_loan_predictions_credit_score ON loan_predictions(credit_score);

-- Add table comments
COMMENT ON TABLE loan_features IS 'Stores loan application features for credit risk assessment';
COMMENT ON TABLE loan_predictions IS 'Stores comprehensive credit risk predictions including PD, LGD, EAD scores';
COMMENT ON COLUMN loan_predictions.pd_score IS 'Probability of Default (0-1)';
COMMENT ON COLUMN loan_predictions.lgd_linear_score IS 'Loss Given Default - Linear Model';
COMMENT ON COLUMN loan_predictions.lgd_logistic_score IS 'Loss Given Default - Logistic Model (0-1)';
COMMENT ON COLUMN loan_predictions.ead_score IS 'Exposure at Default (0-1)';
COMMENT ON COLUMN loan_predictions.expected_loss IS 'Expected monetary loss calculation';